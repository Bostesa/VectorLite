package storage

import (
	"encoding/binary"
	"hash/fnv"
)

const (
	MagicBytes      = "EDB\x00"
	CurrentVersion  = 2
	HeaderSize      = 256
	RecordMetaSize  = 16 // Hash(8) + Dimension(4) + Reserved(4)
	IndexEntrySize  = 16 // Hash(8) + Offset(8)
)

// Header is the file header (fixed 256 bytes)
type Header struct {
	Magic       [4]byte
	Version     uint32
	Dimension   uint32
	RecordCount uint64
	IndexOffset uint64 // Offset to index section
	DataOffset  uint64 // Offset to data section
	Reserved    [216]byte
}

// IndexEntry is a single entry in the index section
type IndexEntry struct {
	Hash   uint64
	Offset int64
}

// Record is a single cached embedding
type Record struct {
	Hash      uint64
	Dimension uint32
	Reserved  uint32
	Vector    []float32
}

func HashText(text string) uint64 {
	h := fnv.New64a()
	h.Write([]byte(text))
	return h.Sum64()
}

func (h *Header) Encode() []byte {
	buf := make([]byte, HeaderSize)
	copy(buf[0:4], h.Magic[:])
	binary.LittleEndian.PutUint32(buf[4:8], h.Version)
	binary.LittleEndian.PutUint32(buf[8:12], h.Dimension)
	binary.LittleEndian.PutUint64(buf[12:20], h.RecordCount)
	binary.LittleEndian.PutUint64(buf[20:28], h.IndexOffset)
	binary.LittleEndian.PutUint64(buf[28:36], h.DataOffset)
	return buf
}

func DecodeHeader(buf []byte) *Header {
	h := &Header{}
	copy(h.Magic[:], buf[0:4])
	h.Version = binary.LittleEndian.Uint32(buf[4:8])
	h.Dimension = binary.LittleEndian.Uint32(buf[8:12])
	h.RecordCount = binary.LittleEndian.Uint64(buf[12:20])
	h.IndexOffset = binary.LittleEndian.Uint64(buf[20:28])
	h.DataOffset = binary.LittleEndian.Uint64(buf[28:36])
	return h
}

func NewHeader(dimension uint32) *Header {
	h := &Header{
		Version:     CurrentVersion,
		Dimension:   dimension,
		IndexOffset: 0, // Will be set when closing
		DataOffset:  HeaderSize,
	}
	copy(h.Magic[:], MagicBytes)
	return h
}

// EncodeIndexEntry encodes an index entry to bytes
func EncodeIndexEntry(entry IndexEntry) []byte {
	buf := make([]byte, IndexEntrySize)
	binary.LittleEndian.PutUint64(buf[0:8], entry.Hash)
	binary.LittleEndian.PutUint64(buf[8:16], uint64(entry.Offset))
	return buf
}

// DecodeIndexEntry decodes an index entry from bytes
func DecodeIndexEntry(buf []byte) IndexEntry {
	return IndexEntry{
		Hash:   binary.LittleEndian.Uint64(buf[0:8]),
		Offset: int64(binary.LittleEndian.Uint64(buf[8:16])),
	}
}
