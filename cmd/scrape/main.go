package main

import (
	"context"
	scrape "github.com/ahauter/astrolabev2/scrape/github"
	"log/slog"
	"os"
	"os/signal"
	"path"
)

const search_endpoint = "https://api.github.com/search/repositories"
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
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))

	//super_cool_function_that_is_super_cool(language, license)
	if len(os.Args) < 2 {
		logger.Error("not enough command line arguments", "usage", "./scrape <out directory>")
		return
	}

	var scraper scrape.ScraperState
	scraper.Logger = logger
	out_dir := os.Args[1]
	if !exists(out_dir) {
		logger.Info("creating output directory", "path", out_dir)
		os.Mkdir(out_dir, 0755)
	}
	statepath := path.Join(out_dir, state_name)
	if exists(statepath) {
		logger.Info("loading state", "path", statepath)
		err := scraper.Load(statepath)
		if err != nil {
			logger.Error("failed to load state", "error", err)
			logger.Error("exiting due to state load error")
			return
		}
		logger.Info("state loaded", "seen_repos", len(scraper.SeenRepos))
	} else {
		logger.Info("no existing state found, starting fresh")
	}
	ctx, _ := signal.NotifyContext(context.Background(), os.Interrupt)
	logger.Info("starting scraper", "language", language, "license", license)
	err := scraper.Start(ctx, language, license, out_dir)
	if err != nil {
		logger.Error("scraper failed", "error", err)
	}
	logger.Info("saving state", "path", statepath)
	err = scraper.Save(statepath)
	if err != nil {
		logger.Error("failed to save state", "error", err)
		logger.Error("scraper may not be able to recover state on future runs")
	} else {
		logger.Info("state saved successfully")
	}
}
