package main

/*
#include <stdlib.h>
*/
import "C"
import (
	"encoding/json"
	"fmt"
	"unsafe"

	"github.com/Bostesa/VectorLite/pkg/storage"
)

var dbCache = make(map[int]*storage.DB)
var nextHandle = 1

//export OpenDB
func OpenDB(pathCStr *C.char, dimension C.int) C.int {
	path := C.GoString(pathCStr)

	db, err := storage.Open(path, uint32(dimension))
	if err != nil {
		return -1
	}

	handle := nextHandle
	nextHandle++
	dbCache[handle] = db

	return C.int(handle)
}

//export CloseDB
func CloseDB(handle C.int) C.int {
	db, ok := dbCache[int(handle)]
	if !ok {
		return -1
	}

	if err := db.Close(); err != nil {
		return -1
	}

	delete(dbCache, int(handle))
	return 0
}

//export Insert
func Insert(handle C.int, textCStr *C.char, vectorPtr *C.float, vectorLen C.int) C.int {
	db, ok := dbCache[int(handle)]
	if !ok {
		return -1
	}

	text := C.GoString(textCStr)

	vector := (*[1 << 30]float32)(unsafe.Pointer(vectorPtr))[:vectorLen:vectorLen]
	vectorCopy := make([]float32, vectorLen)
	copy(vectorCopy, vector)

	if err := db.Insert(text, vectorCopy); err != nil {
		return -1
	}

	return 0
}

//export Get
func Get(handle C.int, textCStr *C.char, outVectorPtr **C.float, outLen *C.int) C.int {
	db, ok := dbCache[int(handle)]
	if !ok {
		return -1
	}

	text := C.GoString(textCStr)

	vector, err := db.Get(text)
	if err != nil {
		return -1
	}

	cVector := (*C.float)(C.malloc(C.size_t(len(vector)) * C.size_t(unsafe.Sizeof(C.float(0)))))
	goSlice := (*[1 << 30]float32)(unsafe.Pointer(cVector))[:len(vector):len(vector)]
	copy(goSlice, vector)

	*outVectorPtr = cVector
	*outLen = C.int(len(vector))

	return 0
}

//export FindSimilar
func FindSimilar(handle C.int, vectorPtr *C.float, vectorLen C.int, threshold C.float,
	outVectorPtr **C.float, outLen *C.int, outScore *C.float) C.int {

	db, ok := dbCache[int(handle)]
	if !ok {
		return -1
	}

	vector := (*[1 << 30]float32)(unsafe.Pointer(vectorPtr))[:vectorLen:vectorLen]
	vectorCopy := make([]float32, vectorLen)
	copy(vectorCopy, vector)

	result, score, err := db.FindSimilar(vectorCopy, float32(threshold))
	if err != nil {
		return -1
	}

	cVector := (*C.float)(C.malloc(C.size_t(len(result)) * C.size_t(unsafe.Sizeof(C.float(0)))))
	goSlice := (*[1 << 30]float32)(unsafe.Pointer(cVector))[:len(result):len(result)]
	copy(goSlice, result)

	*outVectorPtr = cVector
	*outLen = C.int(len(result))
	*outScore = C.float(score)

	return 0
}

//export GetStats
func GetStats(handle C.int) *C.char {
	db, ok := dbCache[int(handle)]
	if !ok {
		return C.CString("")
	}

	stats := db.Stats()
	jsonBytes, err := json.Marshal(stats)
	if err != nil {
		return C.CString("")
	}

	return C.CString(string(jsonBytes))
}

//export FreeVector
func FreeVector(ptr *C.float) {
	C.free(unsafe.Pointer(ptr))
}

//export FreeString
func FreeString(ptr *C.char) {
	C.free(unsafe.Pointer(ptr))
}

//export GetLastError
func GetLastError() *C.char {
	return C.CString("error")
}

func main() {
	fmt.Println("EmbedCache library loaded")
}
