package github

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path"
	"strings"
	"sync"
)

type RepoMetadata struct {
	Id       int
	Commits  []string
	Complete bool
}

type ScraperState struct {
	ApiKey       string
	SeenRepos    []RepoMetadata
	OutPath      string
	AllowedPaths []string
}

func (s *ScraperState) Allowed(path string) bool {
	for _, p := range s.AllowedPaths {
		if strings.HasSuffix(path, p) {
			return true
		}
	}
	return false
}

func exists(dir string) bool {
	_, err := os.Stat(dir)
	if err == nil {
		return true
	}
	return os.IsExist(err)
}

func (s *ScraperState) RepoComplete(id int) bool {
	for _, repo := range s.SeenRepos {
		if repo.Id == id {
			return repo.Complete
		}
	}
	return false
}

func (s *ScraperState) BranchComplete(id int, sha string) bool {
	for _, repo := range s.SeenRepos {
		if repo.Id == id {
			for _, commit := range repo.Commits {
				if commit == sha {
					return true
				}
			}
		}
	}
	return false
}

const num_workers = 8 // todo make this configurable

const search_endpoint = "https://api.github.com/search/repositories"

func _mkdir(path string) {
	if !exists(path) {
		fmt.Printf("Directory %s not found, creating new directory!\n", path)
		os.Mkdir(path, 0755)
	} else {
		return
	}
}
func _try_fetch(ctx context.Context, client *http.Client, url string, ptr any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	resp, err := client.Do(req)
	if err != nil {
		fmt.Println(err.Error())
		panic("Could not get da body")
	}
	if resp.StatusCode != 200 {
		fmt.Println(resp.Status)
		panic("Could not get da body status is bad")
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	err = json.Unmarshal(body, ptr)
	if err != nil {
		return err
	}
	return nil
}

func (s *ScraperState) treeWorker(
	ctx context.Context,
	root TreeResponse, client *http.Client,
	basePath string,
) {
	for _, entry := range root.Tree {
		select {
		case <-ctx.Done():
			return
		default:
			e_path := path.Join(basePath, entry.Path)
			// tree entry
			if entry.Size != nil {
				var blob BlobResponse
				if !s.Allowed(entry.Path) {
					continue
				}
				err := _try_fetch(ctx, client, entry.URL, &blob)
				if err != nil {
					continue
				}
				os.WriteFile(e_path, []byte(blob.Content), 0755)
				// this is a file
				return
			}
			// this is a directory
			var tree TreeResponse
			err := _try_fetch(ctx, client, entry.URL, &tree)
			if err != nil {
				continue
			}
			_mkdir(e_path)
			s.treeWorker(ctx, tree, client, e_path)
		}
	}
}

func (s *ScraperState) repoWorker(
	ctx context.Context,
	wg *sync.WaitGroup,
	client *http.Client,
	ready chan RepoMetadata,
	recv chan RepoItem,
) {
	defer wg.Done()
	//todo open path and check for our repo
	var completed RepoMetadata
	for {
		ready <- RepoMetadata{Complete: true}
		select {
		case targetRepo := <-recv:
			// list branches
			branchesUrl := strings.Split(targetRepo.BranchesURL, "{")[0]
			repo_path := path.Join(s.OutPath, targetRepo.Name)
			completed = RepoMetadata{
				Complete: false,
				Id:       targetRepo.ID,
			}
			_mkdir(repo_path)
			var branches []BranchInfo
			err := _try_fetch(ctx, client, branchesUrl, &branches)
			if err != nil {
				continue
			}
			for _, commit := range branches {
				var commitResp CommitResponse
				err = _try_fetch(ctx, client, commit.Commit.Sha, &commitResp)
				if err != nil {
					continue
				}
				var treeResp TreeResponse
				err = _try_fetch(ctx, client, commitResp.Commit.Tree.URL, &treeResp)
				if err != nil {
					continue
				}
				sha := commitResp.Commit.Tree.SHA
				commit_path := path.Join(repo_path)
				//check for completion of this particular branch
				if s.BranchComplete(targetRepo.ID, sha) {
					continue
				}

				s.treeWorker(ctx, treeResp, client, commit_path)
				completed.Commits = append(completed.Commits, sha)
			}
			break
		case <-ctx.Done():
			ready <- completed
			return
		}
	}
}

func (s *ScraperState) Start(
	ctx context.Context, lang, lic string,
	outpath string,
) error {
	if outpath != "" {
		s.OutPath = outpath
	}
	client := NewClient("")
	// get the repos
	q := fmt.Sprintf(
		"license:%s language:%s", lic, lang,
	)
	params := url.Values{}
	params.Add("q", q)
	fmt.Println(params)
	u, _ := url.Parse(search_endpoint)
	u.RawQuery = params.Encode()
	fmt.Println(u.String())
	resp, err := client.Get(u.String())
	// TODO handle error more gracefully? we're already retrying so maybe this is ok
	if err != nil {
		fmt.Println(err.Error())
		panic("Search unavaible, cannot restart scraping")
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		fmt.Println(resp.Status)
		panic("Could not get da body status is bad")
	}
	var repoInfo RepoSearchResponse
	body, _ := io.ReadAll(resp.Body)
	err = json.Unmarshal(body, &repoInfo)
	if err != nil {
		panic("Could not search for new repositories, try again later")
	}

	ready := make(chan RepoMetadata, num_workers)
	trans := make(chan RepoItem, num_workers)
	var wg sync.WaitGroup
	for i := 0; i < num_workers; i++ {
		wg.Add(1)
		go s.repoWorker(ctx, &wg, &client, ready, trans)
	}
	for _, repoItem := range repoInfo.Items {
		select {
		case <-ready:
			trans <- repoItem
			break
		case <-ctx.Done():
			wg.Wait()
		}
	}
	// check against already scraped repos
	// check the against seen commits
	//spawn workers to download each tree
	return nil
}

func (s *ScraperState) Load(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	err = json.Unmarshal(data, s)
	if s.AllowedPaths == nil {
		s.AllowedPaths = []string{".go", ".mod", ".sum", ".py"}
	}
	if err != nil {
		return err
	}
	return nil
}

func (s *ScraperState) Save(path string) error {
	raw_data, err := json.Marshal(s)
	if err != nil {
		return err
	}
	err = os.WriteFile(path, raw_data, 0644)
	return err
}
