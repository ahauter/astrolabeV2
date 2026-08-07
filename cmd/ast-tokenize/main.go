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
	"encoding/json"
	"runtime/debug"
	"fmt"
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"path"
	"strconv"
	"strings"

	"golang.org/x/tools/go/cfg"
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

// typeCategory is a coarse, risk-relevant classification of a Go type.
type typeCategory string

const (
	typePtr       typeCategory = "ptr"
	typeSlice     typeCategory = "slice"
	typeArray     typeCategory = "array"
	typeMap       typeCategory = "map"
	typeChan      typeCategory = "chan"
	typeInterface typeCategory = "interface"
	typeFunc      typeCategory = "func"
	typeStruct    typeCategory = "struct"
	typeBasic     typeCategory = "basic"
	typeString    typeCategory = "string"
	typeUnknown   typeCategory = "unknown"
	typeLabel     typeCategory = "label"
)

// typeCategoryOf maps a go/types.Type to a coarse category.
// It also accepts a types.Object so that labels can be classified.
func typeCategoryOf(typ types.Type) typeCategory {
	if typ == nil {
		return typeUnknown
	}
	switch t := typ.Underlying().(type) {
	case *types.Pointer:
		return typePtr
	case *types.Slice:
		return typeSlice
	case *types.Array:
		return typeArray
	case *types.Map:
		return typeMap
	case *types.Chan:
		return typeChan
	case *types.Interface:
		return typeInterface
	case *types.Signature:
		return typeFunc
	case *types.Struct:
		return typeStruct
	case *types.Basic:
		if t.Kind() == types.String || t.Kind() == types.UntypedString {
			return typeString
		}
		return typeBasic
	}
	return typeUnknown
}

// typeCategoryOfObject maps a go/types.Object to a coarse category.
func typeCategoryOfObject(obj types.Object) typeCategory {
	if obj == nil {
		return typeUnknown
	}
	if _, ok := obj.(*types.Label); ok {
		return typeLabel
	}
	return typeCategoryOf(obj.Type())
}

// typeCategoryFromExpr infers a coarse type category from an AST type expression.
// Used as a fallback when go/types type-checking is unavailable or fails.
func typeCategoryFromExpr(expr ast.Expr) typeCategory {
	switch e := expr.(type) {
	case *ast.StarExpr:
		return typePtr
	case *ast.ArrayType:
		if e.Len == nil {
			return typeSlice
		}
		return typeArray
	case *ast.MapType:
		return typeMap
	case *ast.ChanType:
		return typeChan
	case *ast.InterfaceType:
		return typeInterface
	case *ast.FuncType:
		return typeFunc
	case *ast.StructType:
		return typeStruct
	case *ast.Ident:
		// Recognize a few common type identifiers without type-checking.
		switch e.Name {
		case "string":
			return typeString
		case "int", "int8", "int16", "int32", "int64",
			"uint", "uint8", "uint16", "uint32", "uint64", "uintptr",
			"float32", "float64", "complex64", "complex128",
			"bool", "byte", "rune":
			return typeBasic
		}
	case *ast.SelectorExpr:
		// e.g. bytes.Buffer, time.Time — value/named type; without full
		// type info we cannot distinguish pointer vs value, so leave unknown.
		return typeUnknown
	}
	return typeUnknown
}

// nameEvent records a single identifier definition or use within a function.
type nameEvent struct {
	pos  int    // token position from declaration start (0-indexed)
	kind string // "def" or "use"
	slot int    // NAME slot (0..nameSlots-1), or -1 for OVF/UNK
	name string // original identifier string
}

// funcCtx holds per-function CFG data accumulated during the walk of a function declaration.
type funcCtx struct {
	funcName           string
	nodeToBlock        map[ast.Node]*cfg.Block      // real body AST node → its CFG block
	rpoOrder           []*cfg.Block                 // blocks in reverse post-order
	rpoIndex           map[int32]int                // block.Index → RPO position
	preds              map[int32][]*cfg.Block       // block.Index → predecessor blocks
	blockStartPos      map[int32]int                // block.Index → first token position in declaration
	blockEndPos        map[int32]int                // block.Index → last token position in declaration
	blockStartRecorded map[int32]bool
	nameEvents         []nameEvent
	slotTypes          map[int]typeCategory         // NAME_N slot → coarse type category
	syncEvents         []syncEvent
	goSpawns           []int
	packageSlots       int                          // number of package-level names in nameStack when function started
	recvSlot           int                          // NAME slot of the method receiver, -1 if none
	importSlots        map[int]struct{}             // slots that are import aliases (excluded from shared set)
}

// syncEvent records a synchronization call inside a function body.
type syncEvent struct {
	Start      int    `json:"start"`
	End        int    `json:"end"`
	Kind       string `json:"kind"`
	RecvSlot   int    `json:"recv"`
	Method     string `json:"method"`
}

// cfgAnnotation is the JSON object emitted on "ANN {...}" lines after each function.
type cfgAnnotation struct {
	Func   string                       `json:"func,omitempty"`
	BB     []int                        `json:"bb"`
	Def    []int                        `json:"def,omitempty"`
	Use    []int                        `json:"use,omitempty"`
	Shared []int                        `json:"shared,omitempty"`
	DU     map[string]int               `json:"du,omitempty"`
	Edges  map[string]map[string]string `json:"edges,omitempty"`
	IDom   map[string]int               `json:"idom,omitempty"`
	Types  map[string]string            `json:"types,omitempty"`
	Sync   []syncEvent                  `json:"sync,omitempty"`
	Go     []int                        `json:"go,omitempty"`
}

type emitter struct {
	w               *bufio.Writer
	nameStack       []string
	scopeMarks      []int
	fields          map[string]int
	funcCtx         *funcCtx    // non-nil only inside a function declaration
	tokenCount      int         // tokens emitted from declaration start (valid when funcCtx != nil)
	fset            *token.FileSet
	currentPos      token.Pos
	globalTokCount  int
	tokenLines      []int
	namePosMap      map[int]string // global token index -> original identifier (for NAME_* tokens only)
	typeInfo        *types.Info // nil if type-checking failed or was not performed
	pkgName         string
	atomicPkgs      map[string]bool // local import names that refer to sync/atomic
	importSlots     map[int]struct{} // NAME slots that are import aliases (excluded from shared set)
}

func newEmitter(w *bufio.Writer, fset *token.FileSet, typeInfo *types.Info, pkgName string) *emitter {
	return &emitter{w: w, fields: map[string]int{}, fset: fset, namePosMap: map[int]string{}, typeInfo: typeInfo, pkgName: pkgName, atomicPkgs: map[string]bool{}, importSlots: map[int]struct{}{}}
}

func (e *emitter) emit(tok string) {
	e.w.WriteString(tok)
	e.w.WriteByte('\n')
	line := 1
	if e.fset != nil && e.currentPos.IsValid() {
		line = e.fset.Position(e.currentPos).Line
	}
	e.tokenLines = append(e.tokenLines, line)
	e.globalTokCount++
	if e.funcCtx != nil {
		e.tokenCount++
	}
}

func (e *emitter) enterScope() { e.scopeMarks = append(e.scopeMarks, len(e.nameStack)) }

func (e *emitter) exitScope() {
	if len(e.scopeMarks) == 0 {
		return
	}
	mark := e.scopeMarks[len(e.scopeMarks)-1]
	e.scopeMarks = e.scopeMarks[:len(e.scopeMarks)-1]
	e.nameStack = e.nameStack[:mark]
}

// introduce registers the identifier in the innermost scope, emits its token,
// and emits a coarse TYPE_* token immediately after.
func (e *emitter) introduce(id *ast.Ident, fallbackExpr ...ast.Expr) {
	if id == nil || id.Name == "_" {
		e.emit("NAME_BLANK")
		return
	}
	name := id.Name
	e.nameStack = append(e.nameStack, name)
	idx := len(e.nameStack) - 1
	if idx >= nameSlots {
		e.emit("NAME_OVF")
		if e.funcCtx != nil {
			e.funcCtx.nameEvents = append(e.funcCtx.nameEvents, nameEvent{e.tokenCount - 1, "def", -1, name})
		}
		return
	}
	e.emit("NAME_" + strconv.Itoa(idx))
	e.namePosMap[e.globalTokCount-1] = name
	if e.funcCtx != nil {
		e.funcCtx.nameEvents = append(e.funcCtx.nameEvents, nameEvent{e.tokenCount - 1, "def", idx, name})
	}

	// Record coarse type category for risk analysis (not emitted as a token).
	cat := e.typeCategoryOfIdent(id)
	if cat == typeUnknown && len(fallbackExpr) > 0 {
		cat = typeCategoryFromExpr(fallbackExpr[0])
	}
	if e.funcCtx != nil && idx < nameSlots {
		if e.funcCtx.slotTypes == nil {
			e.funcCtx.slotTypes = make(map[int]typeCategory)
		}
		e.funcCtx.slotTypes[idx] = cat
	}
}

// syncMethodKind returns the synchronization kind for common sync primitive
// method names, or "" if the method is not a tracked synchronization call.
func syncMethodKind(name string) string {
	switch name {
	case "Lock":
		return "lock"
	case "Unlock":
		return "unlock"
	case "RLock":
		return "rlock"
	case "RUnlock":
		return "runlock"
	case "Add", "Done":
		return "wg_" + strings.ToLower(name)
	case "Wait":
		return "wait"
	case "Signal":
		return "cond_signal"
	case "Broadcast":
		return "cond_broadcast"
	}
	return ""
}

// slotOfName returns the current NAME_N slot for name in the active scope chain,
// or -1 if the name is not bound or overflows the slot limit.
func (e *emitter) slotOfName(name string) int {
	if name == "_" {
		return -1
	}
	for i := len(e.nameStack) - 1; i >= 0; i-- {
		if e.nameStack[i] == name {
			if i >= nameSlots {
				return -1
			}
			return i
		}
	}
	return -1
}

// recvTypeString returns a short string representation of a receiver type for
// function identity metadata.
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

// leftmostIdent returns the leftmost identifier in a selector expression chain,
// e.g. "c.mu" -> "c". Returns nil if the expression is not a name chain.
func leftmostIdent(expr ast.Expr) *ast.Ident {
	switch e := expr.(type) {
	case *ast.Ident:
		return e
	case *ast.SelectorExpr:
		return leftmostIdent(e.X)
	case *ast.StarExpr:
		return leftmostIdent(e.X)
	}
	return nil
}

// typeCategoryOfIdent resolves the coarse type category of an identifier definition
// using go/types when available, otherwise returns unknown.
func (e *emitter) typeCategoryOfIdent(id *ast.Ident) typeCategory {
	if e.typeInfo == nil {
		return typeUnknown
	}
	obj := e.typeInfo.Defs[id]
	if obj == nil {
		return typeUnknown
	}
	return typeCategoryOfObject(obj)
}

// register adds name to the innermost scope without emitting. Used during the
// package-scope pre-pass so later references can resolve to a NAME slot.
// Returns the NAME slot assigned, or -1 if the name was skipped.
func (e *emitter) register(name string) int {
	if name == "" || name == "_" {
		return -1
	}
	idx := len(e.nameStack)
	e.nameStack = append(e.nameStack, name)
	return idx
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
				if e.funcCtx != nil {
					e.funcCtx.nameEvents = append(e.funcCtx.nameEvents, nameEvent{e.tokenCount - 1, "use", -1, name})
				}
				return
			}
			e.emit("NAME_" + strconv.Itoa(i))
			e.namePosMap[e.globalTokCount-1] = name
			if e.funcCtx != nil {
				e.funcCtx.nameEvents = append(e.funcCtx.nameEvents, nameEvent{e.tokenCount - 1, "use", i, name})
			}
			return
		}
	}
	e.emit("NAME_UNK")
	if e.funcCtx != nil {
		e.funcCtx.nameEvents = append(e.funcCtx.nameEvents, nameEvent{e.tokenCount - 1, "use", -2, name})
	}
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
				e.introduce(n, f.Type)
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

// buildFuncCtx constructs a funcCtx for the given function body.
func buildFuncCtx(g *cfg.CFG, body *ast.BlockStmt) *funcCtx {
	// Collect real AST node pointers (excludes synthetic nodes injected by cfg.New).
	realNodes := make(map[ast.Node]bool)
	ast.Inspect(body, func(n ast.Node) bool {
		if n != nil {
			realNodes[n] = true
		}
		return true
	})

	// Compute RPO via iterative DFS from the entry block.
	type frame struct {
		blk     *cfg.Block
		succIdx int
	}
	visited := make(map[int32]bool, len(g.Blocks))
	postOrder := make([]*cfg.Block, 0, len(g.Blocks))
	stack := []frame{{g.Blocks[0], 0}}
	visited[g.Blocks[0].Index] = true
	for len(stack) > 0 {
		top := &stack[len(stack)-1]
		if top.succIdx < len(top.blk.Succs) {
			s := top.blk.Succs[top.succIdx]
			top.succIdx++
			if !visited[s.Index] {
				visited[s.Index] = true
				stack = append(stack, frame{s, 0})
			}
		} else {
			postOrder = append(postOrder, top.blk)
			stack = stack[:len(stack)-1]
		}
	}
	rpoOrder := make([]*cfg.Block, len(postOrder))
	for i, b := range postOrder {
		rpoOrder[len(postOrder)-1-i] = b
	}
	rpoIndex := make(map[int32]int, len(rpoOrder))
	for i, b := range rpoOrder {
		rpoIndex[b.Index] = i
	}

	// Build predecessor map from successor edges.
	preds := make(map[int32][]*cfg.Block, len(g.Blocks))
	for _, blk := range g.Blocks {
		for _, succ := range blk.Succs {
			preds[succ.Index] = append(preds[succ.Index], blk)
		}
	}

	// Map real AST nodes to their block.
	nodeToBlock := make(map[ast.Node]*cfg.Block)
	for _, blk := range g.Blocks {
		for _, n := range blk.Nodes {
			if realNodes[n] {
				nodeToBlock[n] = blk
			}
		}
	}

	return &funcCtx{
		nodeToBlock:        nodeToBlock,
		rpoOrder:           rpoOrder,
		rpoIndex:           rpoIndex,
		preds:              preds,
		blockStartPos:      make(map[int32]int),
		blockEndPos:        make(map[int32]int),
		blockStartRecorded: make(map[int32]bool),
		slotTypes:          make(map[int]typeCategory),
		syncEvents:         []syncEvent{},
		goSpawns:           []int{},
		recvSlot:           -1,
	}
}

// computeDominance computes immediate dominators using the Cooper/Harvey/Kennedy
// iterative algorithm. Returns a map from block.Index to the block.Index of its
// immediate dominator (-1 for the entry block).
func computeDominance(rpoOrder []*cfg.Block, rpoIndex map[int32]int, preds map[int32][]*cfg.Block) map[int32]int32 {
	if len(rpoOrder) == 0 {
		return nil
	}
	// idom[i] = RPO index of immediate dominator of rpoOrder[i]; -1 = undefined.
	idom := make([]int, len(rpoOrder))
	for i := range idom {
		idom[i] = -1
	}
	idom[0] = 0
	changed := true
	for changed {
		changed = false
		for i := 1; i < len(rpoOrder); i++ {
			b := rpoOrder[i]
			newIdom := -1
			for _, pred := range preds[b.Index] {
				pi, ok := rpoIndex[pred.Index]
				if !ok || idom[pi] == -1 {
					continue
				}
				if newIdom == -1 {
					newIdom = pi
				} else {
					f1, f2 := pi, newIdom
					for f1 != f2 {
						for f1 > f2 {
							f1 = idom[f1]
						}
						for f2 > f1 {
							f2 = idom[f2]
						}
					}
					newIdom = f1
				}
			}
			if newIdom != -1 && idom[i] != newIdom {
				idom[i] = newIdom
				changed = true
			}
		}
	}
	result := make(map[int32]int32, len(rpoOrder))
	for i, b := range rpoOrder {
		if i == 0 {
			result[b.Index] = -1
		} else if idom[i] >= 0 {
			result[b.Index] = rpoOrder[idom[i]].Index
		}
	}
	return result
}

// buildAnnotation computes the cfgAnnotation from an accumulated funcCtx.
func buildAnnotation(ctx *funcCtx) *cfgAnnotation {
	if ctx == nil || len(ctx.rpoOrder) == 0 {
		return nil
	}
	ann := &cfgAnnotation{
		Func:  ctx.funcName,
		DU:    make(map[string]int),
		Edges: make(map[string]map[string]string),
		IDom:  make(map[string]int),
		Sync:  ctx.syncEvents,
		Go:    ctx.goSpawns,
	}

	// BB: block start positions in RPO order.
	for _, blk := range ctx.rpoOrder {
		if pos, ok := ctx.blockStartPos[blk.Index]; ok {
			ann.BB = append(ann.BB, pos)
		}
	}
	if len(ann.BB) == 0 {
		return nil
	}

	// Def/Use events.
	for _, ev := range ctx.nameEvents {
		if ev.kind == "def" {
			ann.Def = append(ann.Def, ev.pos)
		} else {
			ann.Use = append(ann.Use, ev.pos)
		}
	}

	// Shared: package-level variables, receiver identifiers, and unknown/unresolved
	// identifiers. Local variables (function parameters and in-scope declarations)
	// and import aliases are intentionally excluded.
	for _, ev := range ctx.nameEvents {
		if ev.kind != "use" {
			continue
		}
		if ev.slot >= 0 {
			if _, isImport := ctx.importSlots[ev.slot]; isImport {
				continue
			}
		}
		isShared := false
		switch {
		case ev.slot < 0:
			// NAME_OVF (-1) and NAME_UNK (-2): conservatively treat as shared.
			isShared = true
		case ctx.recvSlot >= 0 && ev.slot == ctx.recvSlot:
			// Method receiver: the struct it points to may be shared.
			isShared = true
		case ev.slot < ctx.packageSlots:
			// Package-level variable, function, type, or constant.
			isShared = true
		}
		if isShared {
			ann.Shared = append(ann.Shared, ev.pos)
		}
	}

	// DU chains: for each use, find the most recent def with same (slot, name).
	for _, useEv := range ctx.nameEvents {
		if useEv.kind != "use" || useEv.slot < 0 {
			continue
		}
		bestPos := -1
		for _, defEv := range ctx.nameEvents {
			if defEv.kind != "def" || defEv.slot != useEv.slot || defEv.name != useEv.name {
				continue
			}
			if defEv.pos < useEv.pos && defEv.pos > bestPos {
				bestPos = defEv.pos
			}
		}
		if bestPos >= 0 {
			ann.DU[strconv.Itoa(useEv.pos)] = bestPos
		}
	}

	// Edges: block_exit_pos → {successor_entry_pos: edge_type}.
	for _, blk := range ctx.rpoOrder {
		exitPos, hasExit := ctx.blockEndPos[blk.Index]
		if !hasExit || len(blk.Succs) == 0 {
			continue
		}
		edgeMap := make(map[string]string)
		for si, succ := range blk.Succs {
			entryPos, hasEntry := ctx.blockStartPos[succ.Index]
			if !hasEntry {
				continue
			}
			var et string
			switch {
			case ctx.rpoIndex[succ.Index] < ctx.rpoIndex[blk.Index]:
				et = "B"
			case len(blk.Succs) == 2 && si == 0:
				et = "T"
			case len(blk.Succs) == 2:
				et = "F"
			default:
				et = "U"
			}
			edgeMap[strconv.Itoa(entryPos)] = et
		}
		if len(edgeMap) > 0 {
			ann.Edges[strconv.Itoa(exitPos)] = edgeMap
		}
	}

	// IDom: compute and store immediate dominators keyed by block.Index.
	idom := computeDominance(ctx.rpoOrder, ctx.rpoIndex, ctx.preds)
	for blockIdx, domIdx := range idom {
		ann.IDom[strconv.Itoa(int(blockIdx))] = int(domIdx)
	}

	// Types: coarse category for each NAME_N slot introduced in the function.
	if len(ctx.slotTypes) > 0 {
		ann.Types = make(map[string]string)
		for slot, cat := range ctx.slotTypes {
			ann.Types[strconv.Itoa(slot)] = string(cat)
		}
	}

	return ann
}

// cfgNodeBefore silently records the token position at which we begin walking
// an AST node. If the node is the first in its CFG block, records the block
// start position.
func (e *emitter) cfgNodeBefore(node ast.Node) {
	if e.funcCtx == nil || node == nil {
		return
	}
	blk, ok := e.funcCtx.nodeToBlock[node]
	if !ok {
		return
	}
	if !e.funcCtx.blockStartRecorded[blk.Index] {
		e.funcCtx.blockStartPos[blk.Index] = e.tokenCount
		e.funcCtx.blockStartRecorded[blk.Index] = true
	}
}

// cfgNodeAfter silently records the token position after walking an AST node.
// If the node is the last real node in its CFG block, records the block end position.
func (e *emitter) cfgNodeAfter(node ast.Node) {
	if e.funcCtx == nil || node == nil {
		return
	}
	blk, ok := e.funcCtx.nodeToBlock[node]
	if !ok {
		return
	}
	var lastReal ast.Node
	for _, n := range blk.Nodes {
		if _, isReal := e.funcCtx.nodeToBlock[n]; isReal {
			lastReal = n
		}
	}
	if node == lastReal {
		e.funcCtx.blockEndPos[blk.Index] = e.tokenCount - 1
	}
}

// emitANN marshals the funcCtx into a JSON annotation line and writes it
// directly to the output buffer (bypassing emit() so tokenCount is unaffected).
func (e *emitter) emitANN() {
	if e.funcCtx == nil {
		return
	}
	ann := buildAnnotation(e.funcCtx)
	if ann == nil {
		return
	}
	b, err := json.Marshal(ann)
	if err != nil {
		return
	}
	e.w.WriteString("ANN ")
	e.w.Write(b)
	e.w.WriteByte('\n')
}

// walk is the core dispatch over AST node kinds.
func (e *emitter) walk(n ast.Node) {
	if n == nil {
		return
	}
	if n.Pos().IsValid() {
		e.currentPos = n.Pos()
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
		callStart := e.tokenCount
		e.emit("OPEN_CALL")

		// Detect synchronization primitive calls for race analysis.
		var pendingSync *syncEvent
		if e.funcCtx != nil {
			if sel, ok := v.Fun.(*ast.SelectorExpr); ok && sel.Sel != nil {
				if kind := syncMethodKind(sel.Sel.Name); kind != "" {
					if id := leftmostIdent(sel.X); id != nil {
						pendingSync = &syncEvent{
							Start:    callStart,
							Kind:     kind,
							RecvSlot: e.slotOfName(id.Name),
							Method:   sel.Sel.Name,
						}
					}
				}
				// sync/atomic package calls (e.g. atomic.AddInt64, atomic.Load).
				if pendingSync == nil && sel.Sel != nil {
					if id, ok := sel.X.(*ast.Ident); ok && id != nil && e.atomicPkgs[id.Name] {
						pendingSync = &syncEvent{
							Start:    callStart,
							Kind:     "atomic",
							RecvSlot: -1,
							Method:   sel.Sel.Name,
						}
					}
				}
			}
		}

		e.walk(v.Fun)
		for _, a := range v.Args {
			e.walk(a)
		}
		if v.Ellipsis.IsValid() {
			e.emit("ELLIPSIS")
		}
		e.emit("CLOSE_CALL")

		if pendingSync != nil && e.funcCtx != nil {
			pendingSync.End = e.tokenCount - 1
			e.funcCtx.syncEvents = append(e.funcCtx.syncEvents, *pendingSync)
		}
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
		outerCtx := e.funcCtx
		outerCount := e.tokenCount
		e.tokenCount = 0
		e.funcCtx = buildFuncCtx(cfg.New(v.Body, func(*ast.CallExpr) bool { return true }), v.Body)
		if e.funcCtx != nil {
			e.funcCtx.packageSlots = len(e.nameStack)
			e.funcCtx.importSlots = e.importSlots
		}
		e.emit("OPEN_FUNC_LIT")
		e.enterScope()
		e.walkFuncType(v.Type)
		e.walk(v.Body)
		e.exitScope()
		e.emit("CLOSE_FUNC_LIT")
		e.emitANN()
		e.funcCtx = outerCtx
		e.tokenCount = outerCount
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
			e.cfgNodeBefore(s)
			e.walk(s)
			e.cfgNodeAfter(s)
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
					e.introduce(id)
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
			e.cfgNodeBefore(v.Init)
			e.walk(v.Init)
			e.cfgNodeAfter(v.Init)
		}
		e.cfgNodeBefore(v.Cond)
		e.walk(v.Cond)
		e.cfgNodeAfter(v.Cond)
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
			e.cfgNodeBefore(v.Init)
			e.walk(v.Init)
			e.cfgNodeAfter(v.Init)
		}
		if v.Cond != nil {
			e.cfgNodeBefore(v.Cond)
			e.walk(v.Cond)
			e.cfgNodeAfter(v.Cond)
		}
		if v.Post != nil {
			e.cfgNodeBefore(v.Post)
			e.walk(v.Post)
			e.cfgNodeAfter(v.Post)
		}
		e.walk(v.Body)
		e.exitScope()
		e.emit("CLOSE_FOR")
	case *ast.RangeStmt:
		e.emit("OPEN_RANGE")
		e.enterScope()
		if v.Tok == token.DEFINE {
			if id, ok := v.Key.(*ast.Ident); ok && id != nil {
				e.introduce(id)
			}
			if id, ok := v.Value.(*ast.Ident); ok && id != nil {
				e.introduce(id)
			}
		} else {
			e.walk(v.Key)
			e.walk(v.Value)
		}
		e.cfgNodeBefore(v.X)
		e.walk(v.X)
		e.cfgNodeAfter(v.X)
		e.walk(v.Body)
		e.exitScope()
		e.emit("CLOSE_RANGE")
	case *ast.SwitchStmt:
		e.emit("OPEN_SWITCH")
		e.enterScope()
		if v.Init != nil {
			e.cfgNodeBefore(v.Init)
			e.walk(v.Init)
			e.cfgNodeAfter(v.Init)
		}
		if v.Tag != nil {
			e.cfgNodeBefore(v.Tag)
			e.walk(v.Tag)
			e.cfgNodeAfter(v.Tag)
		}
		e.walk(v.Body)
		e.exitScope()
		e.emit("CLOSE_SWITCH")
	case *ast.TypeSwitchStmt:
		e.emit("OPEN_TYPE_SWITCH")
		e.enterScope()
		if v.Init != nil {
			e.cfgNodeBefore(v.Init)
			e.walk(v.Init)
			e.cfgNodeAfter(v.Init)
		}
		e.cfgNodeBefore(v.Assign)
		e.walk(v.Assign)
		e.cfgNodeAfter(v.Assign)
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
			e.cfgNodeBefore(s)
			e.walk(s)
			e.cfgNodeAfter(s)
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
			e.cfgNodeBefore(v.Comm)
			e.walk(v.Comm)
			e.cfgNodeAfter(v.Comm)
		}
		for _, s := range v.Body {
			e.cfgNodeBefore(s)
			e.walk(s)
			e.cfgNodeAfter(s)
		}
		e.exitScope()
		e.emit("CLOSE_COMM_CLAUSE")
	case *ast.GoStmt:
		goPos := e.tokenCount
		e.emit("OPEN_GO")
		if e.funcCtx != nil {
			e.funcCtx.goSpawns = append(e.funcCtx.goSpawns, goPos)
		}
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
		e.introduce(v.Label)
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
		e.cfgNodeBefore(v)
		for _, n := range v.Names {
			// package-level names were pre-registered; inner var/const in
			// function bodies were not, so if not found, introduce fresh.
			if !e.nameInStack(n.Name) {
				e.introduce(n, v.Type)
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
		e.cfgNodeAfter(v)
		e.emit("CLOSE_VALUE_SPEC")
	case *ast.TypeSpec:
		e.emit("OPEN_TYPE_SPEC")
		if !e.nameInStack(v.Name.Name) {
			e.introduce(v.Name, v.Type)
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
		outerCtx := e.funcCtx
		outerCount := e.tokenCount
		e.tokenCount = 0
		if v.Body != nil {
			e.funcCtx = buildFuncCtx(cfg.New(v.Body, func(*ast.CallExpr) bool { return true }), v.Body)
		}
		if e.funcCtx != nil {
			funcKey := e.pkgName + "." + v.Name.Name
			if v.Recv != nil && len(v.Recv.List) > 0 {
				funcKey = e.pkgName + ".(" + recvTypeString(v.Recv.List[0].Type) + ")." + v.Name.Name
			}
			e.funcCtx.funcName = funcKey
			e.funcCtx.packageSlots = len(e.nameStack)
			e.funcCtx.importSlots = e.importSlots
		}
		e.emit("OPEN_FUNC_DECL")
		e.enterScope()
		if v.Recv != nil {
			e.emit("OPEN_RECV")
			e.walkFieldList(v.Recv, introduceNames)
			if e.funcCtx != nil && len(v.Recv.List) > 0 && len(v.Recv.List[0].Names) > 0 {
				e.funcCtx.recvSlot = e.slotOfName(v.Recv.List[0].Names[0].Name)
			}
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
		e.emitANN()
		e.funcCtx = outerCtx
		e.tokenCount = outerCount
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
					if s.Path == nil {
						continue
					}
					local := ""
					base := extractImportName(s.Path.Value)
					if s.Name != nil {
						if s.Name.Name != "." && s.Name.Name != "_" {
							local = s.Name.Name
						}
					} else {
						local = base
					}
					if local != "" {
						slot := e.register(local)
						if slot >= 0 {
							e.importSlots[slot] = struct{}{}
						}
						if base == "atomic" {
							e.atomicPkgs[local] = true
						}
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

// typeCheckFile attempts to type-check a single file as its own package.
// It returns the partially populated types.Info even when type-checking
// reports errors (e.g., unused variables), because the type info is still
// useful for the coarse type categories we emit.
func typeCheckFile(fset *token.FileSet, file *ast.File) *types.Info {
	pkg := types.NewPackage(file.Name.Name, file.Name.Name)
	info := &types.Info{
		Types: make(map[ast.Expr]types.TypeAndValue),
		Defs:  make(map[*ast.Ident]types.Object),
		Uses:  make(map[*ast.Ident]types.Object),
	}
	conf := types.Config{
		Importer:    importer.Default(),
		Error:       func(err error) { /* ignore type errors; partial info is still useful */ },
		FakeImportC: true,
	}
	conf.Check(pkg.Path(), fset, []*ast.File{file}, info)
	return info
}

func tokenizeFile(path string, w *bufio.Writer) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("panic: %v\n%s", r, debug.Stack())
		}
	}()

	fset := token.NewFileSet()
	src, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	file, err := parser.ParseFile(fset, path, src, parser.SkipObjectResolution)
	if err != nil {
		return err
	}
	typeInfo := typeCheckFile(fset, file)
	pkgName := file.Name.Name
	e := newEmitter(w, fset, typeInfo, pkgName)
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

	// Emit position map so downstream tools can map token indices to lines.
	if len(e.tokenLines) > 0 {
		b, _ := json.Marshal(e.tokenLines)
		e.w.WriteString("POSMAP ")
		e.w.Write(b)
		e.w.WriteByte('\n')
	}

	// Emit package import short names so downstream tools can filter them out.
	var importNames []string
	for _, imp := range file.Imports {
		if imp.Name != nil && imp.Name.Name != "." && imp.Name.Name != "_" {
			importNames = append(importNames, imp.Name.Name)
		} else {
			importNames = append(importNames, extractImportName(imp.Path.Value))
		}
	}
	if len(importNames) > 0 {
		b, _ := json.Marshal(importNames)
		e.w.WriteString("PKGS ")
		e.w.Write(b)
		e.w.WriteByte('\n')
	}

	// Emit NAMEPOSMAP so downstream tools can map each NAME_* token position
	// back to its original identifier string (works across multiple functions).
	if len(e.namePosMap) > 0 {
		b, _ := json.Marshal(e.namePosMap)
		e.w.WriteString("NAMEPOSMAP ")
		e.w.Write(b)
		e.w.WriteByte('\n')
	}
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
