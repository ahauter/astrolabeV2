package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"

	scrape "github.com/ahauter/astrolabev2/scrape/github"
)

const search_endpoint = "https://api.github.com/search/repositories"

// TODO previously visited repos skip
func findRepos() {
	lang := "golang"
	license := "mit"
	limit := "1"
	q := fmt.Sprintf(
		"license:%s language:%s limit:", license, lang,
	)
	params := url.Values{}
	params.Add("q", q)
	params.Add("per_page", limit)
	params.Add("page", limit)
	fmt.Println(params)
	u, _ := url.Parse(search_endpoint)
	u.RawQuery = params.Encode()
	fmt.Println(u.String())
	resp, _ := http.Get(u.String())
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	var results scrape.RepoSearchResponse
	json.Unmarshal(body, &results)
	fmt.Println(len(body))
	fmt.Println(string(body))
	fmt.Println(results.TotalCount)
	fmt.Println(results.Items[0].BranchesURL)
	fmt.Println(results.Items[0].TreesURL)
	new_url := strings.Split(results.Items[0].BranchesURL, "{")[0]
	fmt.Println(new_url)
	resp2, _ := http.Get(new_url)
	defer resp2.Body.Close()
	body2, _ := io.ReadAll(resp2.Body)
	fmt.Println(string(body2))
}

func main() {
	fmt.Println("Hello world")
	fmt.Println(os.Args[0])
	findRepos()
}
