package main

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path"
	"strings"

	scrape "github.com/ahauter/astrolabev2/scrape/github"
)

const search_endpoint = "https://api.github.com/search/repositories"

func getDaBody(url string) []byte {
	resp, err := http.Get(url)
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
		fmt.Println(err.Error())
		panic("Could not read da body")
	}
	return body
}

func getDaJSON(url string, ptr any) {
	err := json.Unmarshal(getDaBody(url), ptr)
	if err != nil {
		fmt.Println(err.Error())
		panic("Could not do da jsons")
	}
}

func ContentStr(s *scrape.BlobResponse) string {
	res, _ := base64.RawStdEncoding.DecodeString(s.Content)
	return string(res)
}

// TODO previously visited repos skip
func super_cool_function_that_is_super_cool(lang, license string) {
	q := fmt.Sprintf(
		"license:%s language:%s", license, lang,
	)
	params := url.Values{}
	params.Add("q", q)
	fmt.Println(params)
	u, _ := url.Parse(search_endpoint)
	u.RawQuery = params.Encode()
	fmt.Println(u.String())
	var results scrape.RepoSearchResponse
	getDaJSON(u.String(), &results)
	fmt.Println(results.Items[0].BranchesURL)
	fmt.Println(results.Items[0].TreesURL)
	new_url := strings.Split(results.Items[0].BranchesURL, "{")[0]
	var results2 []scrape.BranchInfo
	getDaJSON(new_url, &results2)
	newest_url := results2[0].Commit.Url
	var commitResp scrape.CommitResponse
	getDaJSON(newest_url, &commitResp)
	newesest_url := commitResp.Commit.Tree.URL
	var results_super_good_this_time scrape.TreeResponse
	getDaJSON(newesest_url, &results_super_good_this_time)
	even_bestest_url_ever := results_super_good_this_time.Tree[0].URL
	fmt.Println(even_bestest_url_ever)
	fmt.Println(results_super_good_this_time.Tree[0].Path)
	var the_coolest_object scrape.BlobResponse
	getDaJSON(even_bestest_url_ever, &the_coolest_object)
	fmt.Println(ContentStr(&the_coolest_object))
	return
}

type RepoMetadata struct {
	Id      string
	Commits []string
}

type ScraperState struct {
	SeenRepos []string
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

func exists(dir string) bool {
	_, err := os.Stat(dir)
	if err == nil {
		return true
	}
	return os.IsExist(err)
}

const state_name = "state.json"
const language = "go"
const license = "mit"

func main() {
	if len(os.Args) < 2 {
		fmt.Printf("Error! Not enough command line arguments!")
		fmt.Printf("Usage: ./scrape <out directory>")
		return
	}

	var scraper ScraperState
	out_dir := os.Args[1]
	if !exists(out_dir) {
		fmt.Printf("Directory %s not found, creating new directory!\n", out_dir)
		os.Mkdir(out_dir, 0755)
	}
	statepath := path.Join(out_dir, state_name)
	if exists(statepath) {
		err := scraper.Load(statepath)
		fmt.Println(scraper.SeenRepos[0])
		if err != nil {
			fmt.Println(err.Error())
			fmt.Println("Above error occurred while trying to load scraper state; exiting")
			return
		}
	}

	//super_cool_function_that_is_super_cool(language, license)
	scraper.SeenRepos = append(scraper.SeenRepos, "HelloIAmARepoHash")
	err := scraper.Save(statepath)
	if err != nil {
		fmt.Println("Scraper state attempted to save and encountered the following error:")
		fmt.Println(err.Error())
		fmt.Println("Scraper may not be able to recover state on future runs!")
	}
}
