package main

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	scrape "github.com/ahauter/astrolabev2/scrape/github"
)

type commitInfo struct {
	Sha string `json:"sha"`
	Url string `json:"url"`
}

type branchInfo struct {
	Name      string     `json:"name"`
	Commit    commitInfo `json:"commit"`
	Protected bool       `json:"protected"`
}

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
func findRepos(lang, license string) {
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
	var results2 []branchInfo
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
}

func main() {
	findRepos("python", "mit")
}
