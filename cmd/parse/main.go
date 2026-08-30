package main

import (
	"context"
	"fmt"
	"os"

	sitter "github.com/smacker/go-tree-sitter"
	"github.com/smacker/go-tree-sitter/golang"
)

const file_path = "./data/thirdRun/gin/c6d07dcc1ee884cf3298855ce05803637010a085/mode.go"

func getSource() []byte {
	result, err := os.ReadFile(file_path)
	if err != nil {
		panic("Could not read the file!")
	}
	return result
}

func main() {
	fmt.Println("hello world")
	parser := sitter.NewParser()
	parser.SetLanguage(golang.GetLanguage())
	source := getSource()
	tree, err := parser.ParseCtx(context.Background(), nil, source)
	if err != nil {
		panic("Error parsing tree")
	}

	n := tree.RootNode()

	fmt.Println(n)
	fmt.Println(n.ChildCount())
	child := n.NamedChild(10)
	fmt.Println(child.Type())
	fmt.Println(child.StartByte())
	fmt.Println(child.EndByte())
}
