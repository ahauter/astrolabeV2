// ast-tokenize reads Go source files and emits a structural token stream on
// stdout, one token per line. The stream uses open/close pairs for every AST
// node kind, scoped indices for identifiers and fields, and dedicated tokens
// for operators, literal kinds, and predeclared identifiers. String and
// comment contents are dropped.
//
// Usage:
//
//	ast-tokenize file1.go [file2.go ...]     # tokenize listed files
//	ast-tokenize -                           # read newline-separated paths from stdin
//
// Between files (and between top-level declarations) an [EOF] token is
// emitted. A fatal parse error on one file causes that file to be skipped
// with a warning on stderr; the process continues with the next file.
package main

import (
	"bufio"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path"
	"strconv"
	"strings"
)

const (
	nameSlots  = 64
	fieldSlots = 64
)

var predeclared = map[string]string{
	"len": "BI_LEN", "cap": "BI_CAP", "panic": "BI_PANIC",
	"recover": "BI_RECOVER", "make": "BI_MAKE", "new": "BI_NEW",
	"append": "BI_APPEND", "copy": "BI_COPY", "delete": "BI_DELETE",
	"close": "BI_CLOSE", "complex": "BI_COMPLEX", "real": "BI_REAL",
	"imag": "BI_IMAG", "print": "BI_PRINT", "println": "BI_PRINTLN",
	"nil": "BI_NIL", "true": "BI_TRUE", "false": "BI_FALSE", "iota": "BI_IOTA",
	"bool": "T_BOOL", "byte": "T_BYTE", "rune": "T_RUNE", "string": "T_STRING",
	"int":  "T_INT", "int8": "T_INT8", "int16": "T_INT16",
	"int32": "T_INT32", "int64": "T_INT64",
	"uint":  "T_UINT", "uint8": "T_UINT8", "uint16": "T_UINT16",
	"uint32": "T_UINT32", "uint64": "T_UINT64", "uintptr": "T_UINTPTR",
	"float32": "T_FLOAT32", "float64": "T_FLOAT64",
	"complex64": "T_COMPLEX64", "complex128": "T_COMPLEX128",
	"error": "T_ERROR", "any": "T_ANY", "comparable": "T_COMPARABLE",
}

type emitter struct {
	w          *bufio.Writer
	nameStack  []string
	scopeMarks []int
	fields     map[string]int
}

func newEmitter(w *bufio.Writer) *emitter {
	return &emitter{w: w, fields: map[string]int{}}
}

func (e *emitter) emit(tok string) { e.w.WriteString(tok); e.w.WriteByte('\n') }

func (e *emitter) enterScope() { e.scopeMarks = append(e.scopeMarks, len(e.nameStack)) }

func (e *emitter) exitScope() {
	if len(e.scopeMarks) == 0 {
		return
	}
	mark := e.scopeMarks[len(e.scopeMarks)-1]
	e.scopeMarks = e.scopeMarks[:len(e.scopeMarks)-1]
	e.nameStack = e.nameStack[:mark]
}

// introduce registers name in the innermost scope and emits its token.
func (e *emitter) introduce(name string) {
	if name == "_" {
		e.emit("NAME_BLANK")
		return
	}
	e.nameStack = append(e.nameStack, name)
	idx := len(e.nameStack) - 1
	if idx >= nameSlots {
		e.emit("NAME_OVF")
		return
	}
	e.emit("NAME_" + strconv.Itoa(idx))
}

// register adds name to the innermost scope without emitting. Used during the
// package-scope pre-pass so later references can resolve to a NAME slot.
func (e *emitter) register(name string) {
	if name == "" || name == "_" {
		return
	}
	e.nameStack = append(e.nameStack, name)
}

// nameInStack reports whether name is currently bound in any active scope.
func (e *emitter) nameInStack(name string) bool {
	if name == "_" {
		return false
	}
	for i := len(e.nameStack) - 1; i >= 0; i-- {
		if e.nameStack[i] == name {
			return true
		}
	}
	return false
}

// reference looks name up the scope chain. Predeclared identifiers bypass the
// scope stack entirely.
func (e *emitter) reference(name string) {
	if name == "_" {
		e.emit("NAME_BLANK")
		return
	}
	if tok, ok := predeclared[name]; ok {
		e.emit(tok)
		return
	}
	for i := len(e.nameStack) - 1; i >= 0; i-- {
		if e.nameStack[i] == name {
			if i >= nameSlots {
				e.emit("NAME_OVF")
				return
			}
			e.emit("NAME_" + strconv.Itoa(i))
			return
		}
	}
	e.emit("NAME_UNK")
}

// field emits a FIELD token using a per-file index space, assigning a fresh
// slot on first appearance of a given field name.
func (e *emitter) field(name string) {
	idx, ok := e.fields[name]
	if !ok {
		idx = len(e.fields)
		e.fields[name] = idx
	}
	if idx >= fieldSlots {
		e.emit("FIELD_OVF")
		return
	}
	e.emit("FIELD_" + strconv.Itoa(idx))
}

// litToken maps a basic literal kind to its content-free token.
func litToken(kind token.Token) string {
	switch kind {
	case token.INT:
		return "INT_LIT"
	case token.FLOAT:
		return "FLOAT_LIT"
	case token.IMAG:
		return "IMAG_LIT"
	case token.CHAR:
		return "CHAR_LIT"
	case token.STRING:
		return "STRING_LIT"
	}
	return "LIT_UNK"
}

// opToken maps an operator token to its stream token. Covers binary, unary,
// and compound assignment forms.
func opToken(op token.Token) string {
	switch op {
	case token.ADD:
		return "OP_ADD"
	case token.SUB:
		return "OP_SUB"
	case token.MUL:
		return "OP_MUL"
	case token.QUO:
		return "OP_QUO"
	case token.REM:
		return "OP_REM"
	case token.AND:
		return "OP_AND"
	case token.OR:
		return "OP_OR"
	case token.XOR:
		return "OP_XOR"
	case token.SHL:
		return "OP_SHL"
	case token.SHR:
		return "OP_SHR"
	case token.AND_NOT:
		return "OP_ANDNOT"
	case token.LAND:
		return "OP_LAND"
	case token.LOR:
		return "OP_LOR"
	case token.ARROW:
		return "OP_ARROW"
	case token.INC:
		return "OP_INC"
	case token.DEC:
		return "OP_DEC"
	case token.EQL:
		return "OP_EQL"
	case token.NEQ:
		return "OP_NEQ"
	case token.LSS:
		return "OP_LSS"
	case token.GTR:
		return "OP_GTR"
	case token.LEQ:
		return "OP_LEQ"
	case token.GEQ:
		return "OP_GEQ"
	case token.NOT:
		return "OP_NOT"
	case token.ASSIGN:
		return "OP_ASSIGN"
	case token.DEFINE:
		return "OP_DEFINE"
	case token.ADD_ASSIGN:
		return "OP_ADD_ASSIGN"
	case token.SUB_ASSIGN:
		return "OP_SUB_ASSIGN"
	case token.MUL_ASSIGN:
		return "OP_MUL_ASSIGN"
	case token.QUO_ASSIGN:
		return "OP_QUO_ASSIGN"
	case token.REM_ASSIGN:
		return "OP_REM_ASSIGN"
	case token.AND_ASSIGN:
		return "OP_AND_ASSIGN"
	case token.OR_ASSIGN:
		return "OP_OR_ASSIGN"
	case token.XOR_ASSIGN:
		return "OP_XOR_ASSIGN"
	case token.SHL_ASSIGN:
		return "OP_SHL_ASSIGN"
	case token.SHR_ASSIGN:
		return "OP_SHR_ASSIGN"
	case token.AND_NOT_ASSIGN:
		return "OP_ANDNOT_ASSIGN"
	}
	return "OP_UNK"
}

func chanDirToken(dir ast.ChanDir) string {
	switch dir {
	case ast.SEND:
		return "CHAN_SEND"
	case ast.RECV:
		return "CHAN_RECV"
	}
	return "CHAN_BI"
}

func branchTokToken(t token.Token) string {
	switch t {
	case token.BREAK:
		return "BR_BREAK"
	case token.CONTINUE:
		return "BR_CONTINUE"
	case token.GOTO:
		return "BR_GOTO"
	case token.FALLTHROUGH:
		return "BR_FALLTHROUGH"
	}
	return "BR_UNK"
}

func genDeclOpen(tok token.Token) (string, string) {
	switch tok {
	case token.VAR:
		return "OPEN_VAR_DECL", "CLOSE_VAR_DECL"
	case token.CONST:
		return "OPEN_CONST_DECL", "CLOSE_CONST_DECL"
	case token.TYPE:
		return "OPEN_TYPE_DECL", "CLOSE_TYPE_DECL"
	}
	return "OPEN_GEN_DECL", "CLOSE_GEN_DECL"
}

// walkFieldList traverses a FieldList. If introduceNames is true, each Name
// within is registered into the current scope (function params, struct
// fields interpreted as fields, etc.). For struct/interface bodies, names
// are treated as fields instead.
func (e *emitter) walkFieldList(fl *ast.FieldList, mode fieldMode) {
	if fl == nil {
		return
	}
	for _, f := range fl.List {
		e.emit("OPEN_FIELD")
		for _, n := range f.Names {
			switch mode {
			case introduceNames:
				e.introduce(n.Name)
			case fieldNames:
				e.field(n.Name)
			default:
				e.reference(n.Name)
			}
		}
		if f.Type != nil {
			e.walk(f.Type)
		}
		e.emit("CLOSE_FIELD")
	}
}

type fieldMode int

const (
	introduceNames fieldMode = iota
	fieldNames
	referenceNames
)

// walk is the core dispatch over AST node kinds.
func (e *emitter) walk(n ast.Node) {
	if n == nil {
		return
	}
	switch v := n.(type) {

	// --- leaves ---
	case *ast.Ident:
		e.reference(v.Name)
	case *ast.BasicLit:
		e.emit(litToken(v.Kind))

	// --- expressions ---
	case *ast.BinaryExpr:
		e.emit("OPEN_BINOP")
		e.walk(v.X)
		e.emit(opToken(v.Op))
		e.walk(v.Y)
		e.emit("CLOSE_BINOP")
	case *ast.UnaryExpr:
		e.emit("OPEN_UNARY")
		e.emit(opToken(v.Op))
		e.walk(v.X)
		e.emit("CLOSE_UNARY")
	case *ast.StarExpr:
		e.emit("OPEN_STAR")
		e.walk(v.X)
		e.emit("CLOSE_STAR")
	case *ast.ParenExpr:
		e.walk(v.X)
	case *ast.SelectorExpr:
		e.emit("OPEN_SELECTOR")
		e.walk(v.X)
		if v.Sel != nil {
			e.field(v.Sel.Name)
		}
		e.emit("CLOSE_SELECTOR")
	case *ast.IndexExpr:
		e.emit("OPEN_INDEX")
		e.walk(v.X)
		e.walk(v.Index)
		e.emit("CLOSE_INDEX")
	case *ast.IndexListExpr:
		e.emit("OPEN_INDEX_LIST")
		e.walk(v.X)
		for _, idx := range v.Indices {
			e.walk(idx)
		}
		e.emit("CLOSE_INDEX_LIST")
	case *ast.SliceExpr:
		e.emit("OPEN_SLICE")
		e.walk(v.X)
		e.walk(v.Low)
		e.walk(v.High)
		e.walk(v.Max)
		e.emit("CLOSE_SLICE")
	case *ast.TypeAssertExpr:
		e.emit("OPEN_TYPE_ASSERT")
		e.walk(v.X)
		e.walk(v.Type)
		e.emit("CLOSE_TYPE_ASSERT")
	case *ast.CallExpr:
		e.emit("OPEN_CALL")
		e.walk(v.Fun)
		for _, a := range v.Args {
			e.walk(a)
		}
		if v.Ellipsis.IsValid() {
			e.emit("ELLIPSIS")
		}
		e.emit("CLOSE_CALL")
	case *ast.KeyValueExpr:
		e.emit("OPEN_KV")
		e.walk(v.Key)
		e.walk(v.Value)
		e.emit("CLOSE_KV")
	case *ast.CompositeLit:
		e.emit("OPEN_COMPOSITE_LIT")
		e.walk(v.Type)
		for _, el := range v.Elts {
			e.walk(el)
		}
		e.emit("CLOSE_COMPOSITE_LIT")
	case *ast.FuncLit:
		e.emit("OPEN_FUNC_LIT")
		e.enterScope()
		e.walkFuncType(v.Type)
		e.walk(v.Body)
		e.exitScope()
		e.emit("CLOSE_FUNC_LIT")
	case *ast.Ellipsis:
		e.emit("OPEN_ELLIPSIS_TYPE")
		e.walk(v.Elt)
		e.emit("CLOSE_ELLIPSIS_TYPE")

	// --- types ---
	case *ast.ArrayType:
		e.emit("OPEN_ARRAY_TYPE")
		e.walk(v.Len)
		e.walk(v.Elt)
		e.emit("CLOSE_ARRAY_TYPE")
	case *ast.MapType:
		e.emit("OPEN_MAP_TYPE")
		e.walk(v.Key)
		e.walk(v.Value)
		e.emit("CLOSE_MAP_TYPE")
	case *ast.ChanType:
		e.emit("OPEN_CHAN_TYPE")
		e.emit(chanDirToken(v.Dir))
		e.walk(v.Value)
		e.emit("CLOSE_CHAN_TYPE")
	case *ast.FuncType:
		e.walkFuncType(v)
	case *ast.StructType:
		e.emit("OPEN_STRUCT_TYPE")
		e.walkFieldList(v.Fields, fieldNames)
		e.emit("CLOSE_STRUCT_TYPE")
	case *ast.InterfaceType:
		e.emit("OPEN_INTERFACE_TYPE")
		e.walkFieldList(v.Methods, fieldNames)
		e.emit("CLOSE_INTERFACE_TYPE")

	// --- statements ---
	case *ast.BlockStmt:
		e.emit("OPEN_BLOCK")
		e.enterScope()
		for _, s := range v.List {
			e.walk(s)
		}
		e.exitScope()
		e.emit("CLOSE_BLOCK")
	case *ast.ExprStmt:
		e.emit("OPEN_EXPR_STMT")
		e.walk(v.X)
		e.emit("CLOSE_EXPR_STMT")
	case *ast.AssignStmt:
		e.emit("OPEN_ASSIGN")
		for _, lhs := range v.Lhs {
			if v.Tok == token.DEFINE {
				if id, ok := lhs.(*ast.Ident); ok {
					e.introduce(id.Name)
					continue
				}
			}
			e.walk(lhs)
		}
		e.emit(opToken(v.Tok))
		for _, rhs := range v.Rhs {
			e.walk(rhs)
		}
		e.emit("CLOSE_ASSIGN")
	case *ast.IncDecStmt:
		e.emit("OPEN_INCDEC")
		e.walk(v.X)
		e.emit(opToken(v.Tok))
		e.emit("CLOSE_INCDEC")
	case *ast.ReturnStmt:
		e.emit("OPEN_RETURN")
		for _, r := range v.Results {
			e.walk(r)
		}
		e.emit("CLOSE_RETURN")
	case *ast.BranchStmt:
		e.emit("OPEN_BRANCH")
		e.emit(branchTokToken(v.Tok))
		if v.Label != nil {
			e.reference(v.Label.Name)
		}
		e.emit("CLOSE_BRANCH")
	case *ast.IfStmt:
		e.emit("OPEN_IF")
		e.enterScope()
		if v.Init != nil {
			e.walk(v.Init)
		}
		e.walk(v.Cond)
		e.walk(v.Body)
		if v.Else != nil {
			e.emit("OPEN_ELSE")
			e.walk(v.Else)
			e.emit("CLOSE_ELSE")
		}
		e.exitScope()
		e.emit("CLOSE_IF")
	case *ast.ForStmt:
		e.emit("OPEN_FOR")
		e.enterScope()
		if v.Init != nil {
			e.walk(v.Init)
		}
		if v.Cond != nil {
			e.walk(v.Cond)
		}
		if v.Post != nil {
			e.walk(v.Post)
		}
		e.walk(v.Body)
		e.exitScope()
		e.emit("CLOSE_FOR")
	case *ast.RangeStmt:
		e.emit("OPEN_RANGE")
		e.enterScope()
		if v.Tok == token.DEFINE {
			if id, ok := v.Key.(*ast.Ident); ok && id != nil {
				e.introduce(id.Name)
			}
			if id, ok := v.Value.(*ast.Ident); ok && id != nil {
				e.introduce(id.Name)
			}
		} else {
			e.walk(v.Key)
			e.walk(v.Value)
		}
		e.walk(v.X)
		e.walk(v.Body)
		e.exitScope()
		e.emit("CLOSE_RANGE")
	case *ast.SwitchStmt:
		e.emit("OPEN_SWITCH")
		e.enterScope()
		if v.Init != nil {
			e.walk(v.Init)
		}
		if v.Tag != nil {
			e.walk(v.Tag)
		}
		e.walk(v.Body)
		e.exitScope()
		e.emit("CLOSE_SWITCH")
	case *ast.TypeSwitchStmt:
		e.emit("OPEN_TYPE_SWITCH")
		e.enterScope()
		if v.Init != nil {
			e.walk(v.Init)
		}
		e.walk(v.Assign)
		e.walk(v.Body)
		e.exitScope()
		e.emit("CLOSE_TYPE_SWITCH")
	case *ast.CaseClause:
		e.emit("OPEN_CASE")
		e.enterScope()
		for _, c := range v.List {
			e.walk(c)
		}
		for _, s := range v.Body {
			e.walk(s)
		}
		e.exitScope()
		e.emit("CLOSE_CASE")
	case *ast.SelectStmt:
		e.emit("OPEN_SELECT")
		e.enterScope()
		e.walk(v.Body)
		e.exitScope()
		e.emit("CLOSE_SELECT")
	case *ast.CommClause:
		e.emit("OPEN_COMM_CLAUSE")
		e.enterScope()
		if v.Comm != nil {
			e.walk(v.Comm)
		}
		for _, s := range v.Body {
			e.walk(s)
		}
		e.exitScope()
		e.emit("CLOSE_COMM_CLAUSE")
	case *ast.GoStmt:
		e.emit("OPEN_GO")
		e.walk(v.Call)
		e.emit("CLOSE_GO")
	case *ast.DeferStmt:
		e.emit("OPEN_DEFER")
		e.walk(v.Call)
		e.emit("CLOSE_DEFER")
	case *ast.SendStmt:
		e.emit("OPEN_SEND")
		e.walk(v.Chan)
		e.walk(v.Value)
		e.emit("CLOSE_SEND")
	case *ast.LabeledStmt:
		e.emit("OPEN_LABELED")
		e.introduce(v.Label.Name)
		e.walk(v.Stmt)
		e.emit("CLOSE_LABELED")
	case *ast.EmptyStmt:
		e.emit("EMPTY_STMT")
	case *ast.DeclStmt:
		e.walk(v.Decl)

	// --- declarations ---
	case *ast.GenDecl:
		open, close := genDeclOpen(v.Tok)
		e.emit(open)
		for _, s := range v.Specs {
			e.walk(s)
		}
		e.emit(close)
	case *ast.ValueSpec:
		e.emit("OPEN_VALUE_SPEC")
		for _, n := range v.Names {
			// package-level names were pre-registered; inner var/const in
			// function bodies were not, so if not found, introduce fresh.
			if !e.nameInStack(n.Name) {
				e.introduce(n.Name)
			} else {
				e.reference(n.Name)
			}
		}
		if v.Type != nil {
			e.walk(v.Type)
		}
		for _, val := range v.Values {
			e.walk(val)
		}
		e.emit("CLOSE_VALUE_SPEC")
	case *ast.TypeSpec:
		e.emit("OPEN_TYPE_SPEC")
		if !e.nameInStack(v.Name.Name) {
			e.introduce(v.Name.Name)
		} else {
			e.reference(v.Name.Name)
		}
		if v.TypeParams != nil {
			e.enterScope()
			e.walkFieldList(v.TypeParams, introduceNames)
			e.walk(v.Type)
			e.exitScope()
		} else {
			e.walk(v.Type)
		}
		e.emit("CLOSE_TYPE_SPEC")

	case *ast.FuncDecl:
		e.emit("OPEN_FUNC_DECL")
		e.enterScope()
		if v.Recv != nil {
			e.emit("OPEN_RECV")
			e.walkFieldList(v.Recv, introduceNames)
			e.emit("CLOSE_RECV")
			// method name is a field-space identifier (same space as selector.Sel)
			e.field(v.Name.Name)
		} else {
			// package-level func; name was registered in the pre-pass
			e.reference(v.Name.Name)
		}
		e.walkFuncType(v.Type)
		if v.Body != nil {
			e.walk(v.Body)
		}
		e.exitScope()
		e.emit("CLOSE_FUNC_DECL")
	}
}

func (e *emitter) walkFuncType(ft *ast.FuncType) {
	e.emit("OPEN_FUNC_TYPE")
	if ft.TypeParams != nil {
		e.emit("OPEN_TYPE_PARAMS")
		e.walkFieldList(ft.TypeParams, introduceNames)
		e.emit("CLOSE_TYPE_PARAMS")
	}
	e.emit("OPEN_PARAMS")
	e.walkFieldList(ft.Params, introduceNames)
	e.emit("CLOSE_PARAMS")
	if ft.Results != nil {
		e.emit("OPEN_RESULTS")
		e.walkFieldList(ft.Results, introduceNames)
		e.emit("CLOSE_RESULTS")
	}
	e.emit("CLOSE_FUNC_TYPE")
}

// extractImportName derives a short name for an import path for use as a
// package-scope identifier. "fmt" -> fmt, "net/http" -> http.
func extractImportName(quotedPath string) string {
	unq, err := strconv.Unquote(quotedPath)
	if err != nil {
		return ""
	}
	base := path.Base(unq)
	// strip version suffix /v2 /v3 ... which yields a useless name
	for strings.HasPrefix(base, "v") {
		rest := base[1:]
		allDigits := rest != ""
		for _, r := range rest {
			if r < '0' || r > '9' {
				allDigits = false
				break
			}
		}
		if !allDigits {
			break
		}
		parent := path.Dir(unq)
		if parent == "." || parent == "/" {
			break
		}
		unq = parent
		base = path.Base(unq)
	}
	return base
}

// collectPackageNames pre-registers every identifier declared at package
// scope (imports, types, vars, consts, funcs) so later references can
// resolve to a NAME slot rather than NAME_UNK.
func (e *emitter) collectPackageNames(file *ast.File) {
	for _, decl := range file.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			if d.Recv == nil && d.Name != nil {
				e.register(d.Name.Name)
			}
		case *ast.GenDecl:
			for _, spec := range d.Specs {
				switch s := spec.(type) {
				case *ast.ImportSpec:
					if s.Name != nil {
						if s.Name.Name != "." && s.Name.Name != "_" {
							e.register(s.Name.Name)
						}
					} else {
						e.register(extractImportName(s.Path.Value))
					}
				case *ast.ValueSpec:
					for _, n := range s.Names {
						e.register(n.Name)
					}
				case *ast.TypeSpec:
					if s.Name != nil {
						e.register(s.Name.Name)
					}
				}
			}
		}
	}
}

func tokenizeFile(path string, w *bufio.Writer) error {
	fset := token.NewFileSet()
	src, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	file, err := parser.ParseFile(fset, path, src, parser.SkipObjectResolution)
	if err != nil {
		return err
	}
	e := newEmitter(w)
	e.emit("BOS")
	e.enterScope() // package scope
	e.collectPackageNames(file)
	for i, decl := range file.Decls {
		if i > 0 {
			e.emit("EOF")
		}
		e.walk(decl)
	}
	e.exitScope()
	e.emit("EOF")
	return nil
}

func readStdinPaths() []string {
	var paths []string
	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line != "" {
			paths = append(paths, line)
		}
	}
	return paths
}

func main() {
	args := os.Args[1:]
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "usage: ast-tokenize <file.go>... | ast-tokenize -")
		os.Exit(2)
	}
	var paths []string
	if len(args) == 1 && args[0] == "-" {
		paths = readStdinPaths()
	} else {
		paths = args
	}
	w := bufio.NewWriterSize(os.Stdout, 1<<16)
	defer w.Flush()
	ok := 0
	for _, p := range paths {
		if err := tokenizeFile(p, w); err != nil {
			fmt.Fprintf(os.Stderr, "skip %s: %v\n", p, err)
			continue
		}
		ok++
	}
	if ok == 0 && len(paths) > 0 {
		os.Exit(1)
	}
}
