// ast-callgraph builds a static call graph for a set of Go source directories
// and emits a JSONL mapping from callee qualified name to its direct callers.
//
// This helper does NOT require a go.mod; it parses source files directly, so it
// works on scraped code trees.
//
// Usage:
//
//	ast-callgraph dir1 [dir2 ...]
//
// Output lines:
//
//	{"func": "pkg.Func", "callers": ["pkg.Caller", ...]}
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// recvTypeString returns a short string representation of a receiver type.
func recvTypeString(expr ast.Expr) string {
	switch t := expr.(type) {
	case *ast.Ident:
		return t.Name
	case *ast.StarExpr:
		return "*" + recvTypeString(t.X)
	case *ast.SelectorExpr:
		return recvTypeString(t.X) + "." + t.Sel.Name
	case *ast.ArrayType:
		return "[]" + recvTypeString(t.Elt)
	}
	return "?"
}

// funcKey returns the qualified key for a function declaration.
func funcKey(pkgName string, fd *ast.FuncDecl) string {
	if fd.Recv != nil && len(fd.Recv.List) > 0 {
		return pkgName + ".(" + recvTypeString(fd.Recv.List[0].Type) + ")." + fd.Name.Name
	}
	return pkgName + "." + fd.Name.Name
}

// collectImports maps import alias -> imported package short name.
func collectImports(file *ast.File) map[string]string {
	imports := make(map[string]string)
	for _, imp := range file.Imports {
		path := strings.Trim(imp.Path.Value, `"`)
		short := filepath.Base(path)
		if imp.Name != nil && imp.Name.Name != "." && imp.Name.Name != "_" {
			imports[imp.Name.Name] = short
		} else {
			imports[short] = short
		}
	}
	return imports
}

// findGoFiles recursively finds non-test .go files under root.
func findGoFiles(root string) ([]string, error) {
	var files []string
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			return nil
		}
		if strings.HasSuffix(path, ".go") && !strings.HasSuffix(path, "_test.go") {
			files = append(files, path)
		}
		return nil
	})
	return files, err
}

// groupByDir groups file paths by their directory.
func groupByDir(files []string) map[string][]string {
	groups := make(map[string][]string)
	for _, f := range files {
		dir := filepath.Dir(f)
		groups[dir] = append(groups[dir], f)
	}
	return groups
}

// callGraphForDir builds caller->callee edges for all .go files in one directory.
func callGraphForDir(dir string, files []string) map[string]map[string]struct{} {
	fset := token.NewFileSet()
	parsed := make(map[string]*ast.File)
	pkgName := ""
	for _, f := range files {
		file, err := parser.ParseFile(fset, f, nil, parser.SkipObjectResolution)
		if err != nil {
			continue
		}
		parsed[f] = file
		if file.Name != nil && pkgName == "" {
			pkgName = file.Name.Name
		}
	}
	if pkgName == "" {
		return nil
	}

	// First pass: collect all defined function keys.
	defs := make(map[string]struct{})
	for _, file := range parsed {
		ast.Inspect(file, func(n ast.Node) bool {
			if fd, ok := n.(*ast.FuncDecl); ok {
				defs[funcKey(pkgName, fd)] = struct{}{}
			}
			return true
		})
	}

	// Collect method names for over-approximation of selector calls.
	methodsByName := make(map[string][]string)
	for key := range defs {
		if i := strings.LastIndex(key, ")."); i >= 0 {
			methodName := key[i+2:]
			methodsByName[methodName] = append(methodsByName[methodName], key)
		}
	}

	edges := make(map[string]map[string]struct{})
	addEdge := func(caller, callee string) {
		if caller == "" || callee == "" || caller == callee {
			return
		}
		if edges[caller] == nil {
			edges[caller] = make(map[string]struct{})
		}
		edges[caller][callee] = struct{}{}
	}

	// Second pass: resolve call sites.
	for _, file := range parsed {
		imports := collectImports(file)
		var stack []string // enclosing function keys

		var visit func(ast.Node) bool
		visit = func(n ast.Node) bool {
			switch v := n.(type) {
		case *ast.FuncDecl:
			key := funcKey(pkgName, v)
			stack = append(stack, key)
			if v.Body != nil {
				ast.Inspect(v.Body, visit)
			}
			stack = stack[:len(stack)-1]
			return false
		case *ast.FuncLit:
			// Anonymous functions are not attributed to any named caller.
			stack = append(stack, "")
			if v.Body != nil {
				ast.Inspect(v.Body, visit)
			}
			stack = stack[:len(stack)-1]
			return false
			case *ast.CallExpr:
				if len(stack) == 0 || stack[len(stack)-1] == "" {
					return true
				}
				caller := stack[len(stack)-1]
				switch fun := v.Fun.(type) {
				case *ast.Ident:
					addEdge(caller, pkgName+"."+fun.Name)
				case *ast.SelectorExpr:
					if id, ok := fun.X.(*ast.Ident); ok {
						if importedPkg, ok2 := imports[id.Name]; ok2 {
							// Cross-package call; best-effort key using imported short name.
							addEdge(caller, importedPkg+"."+fun.Sel.Name)
						} else {
							// Method call on a local value: over-approximate to all
							// methods with this name defined in the package.
							for _, callee := range methodsByName[fun.Sel.Name] {
								addEdge(caller, callee)
							}
						}
					}
				}
			}
			return true
		}
		ast.Inspect(file, visit)
	}

	return edges
}

func main() {
	dirs := os.Args[1:]
	if len(dirs) == 0 {
		fmt.Fprintln(os.Stderr, "usage: ast-callgraph <dir>...")
		os.Exit(2)
	}

	// callee -> set of callers
	callers := make(map[string]map[string]struct{})
	addCaller := func(callee, caller string) {
		if callee == "" || caller == "" {
			return
		}
		if callers[callee] == nil {
			callers[callee] = make(map[string]struct{})
		}
		callers[callee][caller] = struct{}{}
	}

	for _, dir := range dirs {
		files, err := findGoFiles(dir)
		if err != nil {
			fmt.Fprintf(os.Stderr, "skip %s: %v\n", dir, err)
			continue
		}
		groups := groupByDir(files)
		for _, groupFiles := range groups {
			edges := callGraphForDir(filepath.Dir(groupFiles[0]), groupFiles)
			for caller, callees := range edges {
				for callee := range callees {
					addCaller(callee, caller)
				}
			}
		}
	}

	w := bufio.NewWriterSize(os.Stdout, 1<<16)
	defer w.Flush()

	keys := make([]string, 0, len(callers))
	for k := range callers {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, callee := range keys {
		list := make([]string, 0, len(callers[callee]))
		for c := range callers[callee] {
			list = append(list, c)
		}
		sort.Strings(list)
		b, _ := json.Marshal(map[string]any{
			"func":    callee,
			"callers": list,
		})
		w.Write(b)
		w.WriteByte('\n')
	}
}
