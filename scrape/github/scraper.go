package github

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"golang.org/x/sync/errgroup"
	"io"
	"log/slog"
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
	ApiKey       string         `json:"api_key"`
	SeenRepos    []RepoMetadata `json:"seen_repos"`
	OutPath      string         `json:"out_path"`
	AllowedPaths []string       `json:"allowed_paths"`
	Logger       *slog.Logger   `json:"-"`
}

func (s *ScraperState) logger() *slog.Logger {
	if s.Logger != nil {
		return s.Logger
	}
	return slog.Default()
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

const num_workers = 4 // todo make this configurable

const search_endpoint = "https://api.github.com/search/repositories"

func _mkdir(logger *slog.Logger, p string) error {
	if !exists(p) {
		logger.Info("creating directory", "path", p)
		return os.Mkdir(p, 0755)
	}
	return nil
}

func _try_fetch(ctx context.Context, client *http.Client, url string, ptr any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return errors.New("Request status unexpected: " + resp.Status)
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
	logger *slog.Logger,
) error {
	logger.Debug("downloading blob", "url", url)
	var blob BlobResponse
	err := _try_fetch(ctx, client, url, &blob)
	if err != nil {
		logger.Error("blob download failed", "url", url, "error", err)
		return err
	}
	os.WriteFile(path, ContentByte(&blob), 0644)
	logger.Debug("blob downloaded", "path", path, "size", blob.Size)
	return nil
}

func (s *ScraperState) treeWorker(
	ctx context.Context,
	root TreeResponse,
	client *http.Client,
	basePath string,
	logger *slog.Logger,
) error {
	wg, ctx := errgroup.WithContext(ctx)
	wg.SetLimit(4)
	logger = logger.With("base_path", basePath)
	logger.Info("downloading tree", "entries", len(root.Tree))
	for _, entry := range root.Tree {
		select {
		case <-ctx.Done():
			wg.Wait()
			return errors.New("Context is cancelled before tree could complete")
		default:
			e_path := path.Join(basePath, entry.Path)
			if entry.Size == nil {
				// tree entry
				err := _mkdir(logger, e_path)
				if err != nil {
					panic(fmt.Sprintf("Error making directory %s", e_path))
				}
				continue
			}
			if !s.Allowed(entry.Path) {
				continue
			}
			wg.Go(func() error {
				return downloadBlob(
					ctx, client,
					entry.URL, e_path,
					logger.With("path", entry.Path),
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
	logger := s.logger()
	defer wg.Done()
	var completed RepoMetadata
	for targetRepo := range jobs {
		repoLogger := logger.With("repo", targetRepo.Name, "repo_id", targetRepo.ID)
		repoLogger.Info("processing repo")
		branchesUrl := strings.Split(targetRepo.BranchesURL, "{")[0]
		repo_path := path.Join(s.OutPath, targetRepo.Name)
		completed = RepoMetadata{
			Id: targetRepo.ID,
		}
		_mkdir(repoLogger, repo_path)
		var branches []BranchInfo
		err := _try_fetch(ctx, client, branchesUrl, &branches)
		if err != nil {
			repoLogger.Error("failed to fetch branches", "error", err)
			continue
		}
		for _, commit := range branches {
			var commitResp CommitResponse
			err = _try_fetch(ctx, client, commit.Commit.Url, &commitResp)
			if err != nil {
				repoLogger.Warn("failed to fetch commit", "commit_sha", commit.Commit.Sha, "error", err)
				continue
			}
			tree_url, err := url.Parse(commitResp.Commit.Tree.URL)
			if err != nil {
				repoLogger.Warn("failed to parse tree url", "commit_sha", commit.Commit.Sha, "error", err)
				continue
			}
			tree_params := url.Values{}
			tree_params.Add("recursive", "1")
			tree_url.RawQuery = tree_params.Encode()
			var treeResp TreeResponse
			err = _try_fetch(ctx, client, tree_url.String(), &treeResp)
			if err != nil {
				repoLogger.Warn("failed to fetch tree", "commit_sha", commit.Commit.Sha, "error", err)
				continue
			}
			sha := commitResp.Commit.Tree.SHA
			commit_path := path.Join(repo_path, sha)
			if isCommitComplete(targetRepo.ID, sha, snapshot) {
				repoLogger.Info("skipping completed commit", "commit_sha", sha)
				continue
			}

			err = _mkdir(logger, commit_path)
			if err != nil {
				panic("Could not make directory " + commit_path)
			}
			if err := s.treeWorker(ctx, treeResp, client, commit_path, repoLogger.With("commit_sha", sha)); err != nil {
				repoLogger.Error("tree worker failed", "commit_sha", sha, "error", err)
				continue
			}
			completed.Commits = append(completed.Commits, sha)
		}
		select {
		case results <- completed:
			repoLogger.Info("repo finished", "commits_processed", len(completed.Commits))
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
	logger := s.logger()
	if outpath != "" {
		s.OutPath = outpath
	}
	logger.Info("initializing scraper", "out_path", s.OutPath, "language", lang, "license", lic)
	client := NewClient(s.ApiKey, logger)
	// get the repos
	q := fmt.Sprintf(
		"license:%s language:%s", lic, lang,
	)
	params := url.Values{}
	params.Add("q", q)
	u, _ := url.Parse(search_endpoint)
	u.RawQuery = params.Encode()
	logger.Info("searching repositories", "query", q, "url", u.String())
	resp, err := client.Get(u.String())
	// TODO handle error more gracefully? we're already retrying so maybe this is ok
	if err != nil {
		logger.Error("search request failed", "error", err)
		panic("Search unavaible, cannot restart scraping")
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		logger.Error("search returned non-200 status", "status", resp.Status)
		panic("Could not get da body status is bad")
	}
	var repoInfo RepoSearchResponse
	body, _ := io.ReadAll(resp.Body)
	err = json.Unmarshal(body, &repoInfo)
	if err != nil {
		logger.Error("failed to parse search response", "error", err)
		panic("Could not search for new repositories, try again later")
	}
	logger.Info("search completed", "total_count", repoInfo.TotalCount, "incomplete_results", repoInfo.IncompleteResults, "repos_found", len(repoInfo.Items))

	completedSnap := s.getSnapshot()
	results := make(chan RepoMetadata, num_workers)
	jobs := make(chan RepoItem, num_workers)
	var wg sync.WaitGroup
	logger.Info("starting workers", "num_workers", num_workers)
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

	completedCount := 0
	totalCount := len(repoInfo.Items)
	for completed := range results {
		completedCount++
		logger.Info("repo completed",
			"progress", fmt.Sprintf("%d/%d", completedCount, totalCount),
			"repo_id", completed.Id,
			"commits", len(completed.Commits))
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
	logger.Info("all repos completed", "total", completedCount)
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
