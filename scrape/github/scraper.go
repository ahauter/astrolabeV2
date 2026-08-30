package github

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"golang.org/x/sync/errgroup"
	"io"
	"net/http"
	"net/url"
	"os"
	"path"
	"strings"
	"sync"
)

type RepoMetadata struct {
	Id      int
	Commits []string
}

func isCommitComplete(id int, sha string, snapshot map[int]map[string]bool) bool {
	return snapshot[id][sha]
}

func ContentByte(s *BlobResponse) []byte {
	res, _ := base64.StdEncoding.DecodeString(s.Content)
	return res
}

type ScraperState struct {
	ApiKey       string
	SeenRepos    []RepoMetadata
	OutPath      string
	AllowedPaths []string
}

func (s *ScraperState) getSnapshot() map[int]map[string]bool {
	result := make(map[int]map[string]bool)
	for _, repo := range s.SeenRepos {
		repoSnap := make(map[string]bool)
		for _, commit := range repo.Commits {
			repoSnap[commit] = true
		}
		result[repo.Id] = repoSnap
	}
	return result
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
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		fmt.Println(resp.Status)
		return errors.New("Request status unexpected")
	}
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

func downloadBlob(
	ctx context.Context, client *http.Client,
	url string, path string,
) error {
	var blob BlobResponse
	err := _try_fetch(ctx, client, url, &blob)
	if err != nil {
		return err
	}
	os.WriteFile(path, ContentByte(&blob), 0644)
	return err
}

func (s *ScraperState) treeWorker(
	ctx context.Context,
	root TreeResponse,
	client *http.Client,
	basePath string,
) error {
	var wg errgroup.Group
	wg.SetLimit(16)
	for _, entry := range root.Tree {
		select {
		case <-ctx.Done():
			return errors.New("Context is cancelled before tree could complete")
		default:
			e_path := path.Join(basePath, entry.Path)
			if entry.Size == nil {
				// tree entry
				_mkdir(e_path)
				continue
			}
			if !s.Allowed(entry.Path) {
				continue
			}
			wg.Go(func() error {
				return downloadBlob(
					ctx, client,
					entry.URL, e_path,
				)
			})
		}
	}
	return wg.Wait()
}

func (s *ScraperState) repoWorker(
	ctx context.Context,
	wg *sync.WaitGroup,
	client *http.Client,
	results chan RepoMetadata,
	jobs chan RepoItem,
	snapshot map[int]map[string]bool,
) {
	defer wg.Done()
	//todo open path and check for our repo
	var completed RepoMetadata
	for targetRepo := range jobs {
		// list branches
		branchesUrl := strings.Split(targetRepo.BranchesURL, "{")[0]
		repo_path := path.Join(s.OutPath, targetRepo.Name)
		completed = RepoMetadata{
			Id: targetRepo.ID,
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
			tree_url, err := url.Parse(commitResp.Commit.Tree.URL)
			if err != nil {
				continue
			}
			tree_params := url.Values{}
			tree_params.Add("recursive", "1")
			tree_url.RawQuery = tree_params.Encode()
			var treeResp TreeResponse
			err = _try_fetch(ctx, client, tree_url.String(), &treeResp)
			if err != nil {
				continue
			}
			sha := commitResp.Commit.Tree.SHA
			commit_path := path.Join(repo_path)
			//check for completion of this particular branch
			if isCommitComplete(targetRepo.ID, sha, snapshot) {
				continue
			}

			if err := s.treeWorker(ctx, treeResp, client, commit_path); err != nil {
				continue
			}
			completed.Commits = append(completed.Commits, sha)
		}
		select {
		case results <- completed:
		case <-ctx.Done():
			return
		}
	}
}

func (s *ScraperState) Start(
	ctx context.Context,
	lang, lic string,
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

	completedSnap := s.getSnapshot()
	results := make(chan RepoMetadata, num_workers)
	jobs := make(chan RepoItem, num_workers)
	var wg sync.WaitGroup
	for i := 0; i < num_workers; i++ {
		wg.Add(1)
		go s.repoWorker(ctx, &wg, &client, results, jobs, completedSnap)
	}
	go func() {
		defer close(jobs)
		for _, repoItem := range repoInfo.Items {
			select {
			case jobs <- repoItem:
			case <-ctx.Done():
				return
			}
		}
	}()
	go func() {
		wg.Wait()
		close(results)
	}()

	for completed := range results {
		//assume completed > what we've seen
		seen := false
		for i, c := range s.SeenRepos {
			if completed.Id == c.Id {
				s.SeenRepos[i] = completed
				seen = true
			}
		}
		if !seen {
			s.SeenRepos = append(s.SeenRepos, completed)
		}
	}
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
