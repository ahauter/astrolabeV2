package github

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/url"
	"os"
)

type RepoMetadata struct {
	Id      string
	Commits []string
}

type ScraperState struct {
	ApiKey    string
	SeenRepos []RepoMetadata
}

const num_workers = 8 // todo make this configurable

const search_endpoint = "https://api.github.com/search/repositories"

func (s *ScraperState) Start(
	ctx context.Context, lang, lic string,
) error {
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
		panic("Could not get da body")
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		fmt.Println(resp.Status)
		panic("Could not get da body status is bad")
	}
	body, _ := io.ReadAll(resp.Body)
	fmt.Println(string(body))
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
