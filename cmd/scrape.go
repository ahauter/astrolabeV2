package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	scrape "github.com/ahauter/astrolabev2/scrape/github"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"path"
	"strings"
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
	//fmt.Println(results.Items[0].BranchesURL)
	fmt.Println(results.Items[0].TreesURL)
	new_url := strings.Split(results.Items[0].BranchesURL, "{")[0]
	var results2 []scrape.BranchInfo
	getDaJSON(new_url, &results2)
	newest_url := results2[0].Commit.Url
	var commitResp scrape.CommitResponse
	getDaJSON(newest_url, &commitResp)
	newesest_url := commitResp.Commit.Tree.URL
	tree_params := url.Values{}
	tree_params.Add("recursive", "1")
	tree_url, _ := url.Parse(newesest_url)
	tree_url.RawQuery = tree_params.Encode()
	var results_super_good_this_time scrape.TreeResponse
	getDaJSON(tree_url.String(), &results_super_good_this_time)
	for _, tree_item := range results_super_good_this_time.Tree {
		fmt.Println(tree_item.Path)
	}
	even_bestest_url_ever := results_super_good_this_time.Tree[0].URL
	fmt.Println(even_bestest_url_ever)
	fmt.Println(results_super_good_this_time.Tree[0].Path)
	var the_coolest_object scrape.BlobResponse
	getDaJSON(even_bestest_url_ever, &the_coolest_object)
	fmt.Println(ContentStr(&the_coolest_object))
	return
}

const state_name = "state.json"
const language = "go"
const license = "mit"

func exists(dir string) bool {
	_, err := os.Stat(dir)
	if err == nil {
		return true
	}
	return os.IsExist(err)
}
func main() {
	super_cool_function_that_is_super_cool(language, license)
	if len(os.Args) < 2 {
		fmt.Printf("Error! Not enough command line arguments!")
		fmt.Printf("Usage: ./scrape <out directory>")
		return
	}

	var scraper scrape.ScraperState
	out_dir := os.Args[1]
	if !exists(out_dir) {
		fmt.Printf("Directory %s not found, creating new directory!\n", out_dir)
		os.Mkdir(out_dir, 0755)
	}
	statepath := path.Join(out_dir, state_name)
	if exists(statepath) {
		err := scraper.Load(statepath)
		if err != nil {
			fmt.Println(err.Error())
			fmt.Println("Above error occurred while trying to load scraper state; exiting")
			return
		}
	}
	ctx, _ := signal.NotifyContext(context.Background(), os.Interrupt)
	scraper.Start(ctx, language, license, out_dir)
	err := scraper.Save(statepath)
	if err != nil {
		fmt.Println("Scraper state attempted to save and encountered the following error:")
		fmt.Println(err.Error())
		fmt.Println("Scraper may not be able to recover state on future runs!")
	}
}
