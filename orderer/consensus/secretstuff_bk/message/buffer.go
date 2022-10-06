package message

import (
	"encoding/json"
	"log"
	"sync"
)

type Buffer struct {
	requestQueue  []*Request
	requestLocker *sync.RWMutex

	prepareBuffer map[string]*Prepare
	prepareSet    map[string]bool
	prepareLocker *sync.RWMutex

	preCommitSet    map[string]map[Identify]bool
	preCommitState  map[string]bool
	preCommitLocker *sync.RWMutex

	commitSet    map[string]map[Identify]bool
	commitState  map[string]bool
	commitLocker *sync.RWMutex

	decideSet    map[string]map[Identify]bool
	decideState  map[string]bool
	decideLocker *sync.RWMutex

	voteSet      map[string]map[Identify]bool
    voteState    map[string]bool
    voteLocker   *sync.RWMutex

	executeQueue  []*Prepare
	executeLocker *sync.RWMutex

	checkPointBuffer map[string]map[Identify]bool
	checkPointState  map[string]bool
	checkPointLocker *sync.RWMutex
}

func NewBuffer() *Buffer {
	return &Buffer{
		requestQueue:  make([]*Request, 0),
		requestLocker: new(sync.RWMutex),

		prepareBuffer: make(map[string]*Prepare),
		prepareSet:    make(map[string]bool),
		prepareLocker: new(sync.RWMutex),

		preCommitSet:    make(map[string]map[Identify]bool),
		preCommitState:  make(map[string]bool),
		preCommitLocker: new(sync.RWMutex),

		commitSet:    make(map[string]map[Identify]bool),
		commitState:  make(map[string]bool),
		commitLocker: new(sync.RWMutex),

		decideSet:    make(map[string]map[Identify]bool),
		decideState:  make(map[string]bool),
		decideLocker: new(sync.RWMutex),

		voteSet:      make(map[string]map[Identify]bool),
		voteState:    make(map[string]bool),
		voteLocker:   new(sync.RWMutex),

		executeQueue:  make([]*Prepare, 0),
		executeLocker: new(sync.RWMutex),

		checkPointBuffer: make(map[string]map[Identify]bool),
		checkPointState:  make(map[string]bool),
		checkPointLocker: new(sync.RWMutex),
	}
}

// buffer about request
func (b *Buffer) AppendToRequestQueue(req *Request) {
	b.requestLocker.Lock()
	b.requestQueue = append(b.requestQueue, req)
	b.requestLocker.Unlock()
}

func (b *Buffer) BatchRequest() (batch []*Request) {
	batch = make([]*Request, 0)
	b.requestLocker.Lock()
	for _, r := range b.requestQueue {
		batch = append(batch, r)
	}
	b.requestQueue = make([]*Request, 0)
	b.requestLocker.Unlock()
	return
}

func (b *Buffer) SizeofRequestQueue() (l int) {
	b.requestLocker.RLock()
	l = len(b.requestQueue)
	b.requestLocker.RUnlock()
	return
}

// buffer about prepare
func (b *Buffer) BufferPrepareMsg(msg *Prepare) {
	b.prepareLocker.Lock()
	b.prepareBuffer[msg.Digest] = msg
	b.prepareSet[ViewSequenceString(msg.View, msg.Sequence)] = true
	b.prepareLocker.Unlock()
}

func (b *Buffer) ClearPrepareMsg(digest string) {
	b.prepareLocker.Lock()
	msg := b.prepareBuffer[digest]
	delete(b.prepareSet, ViewSequenceString(msg.View, msg.Sequence))
	delete(b.prepareBuffer, digest)
	b.prepareLocker.Unlock()
}

func (b *Buffer) IsExistPrepareMsg(view View, seq Sequence) bool {
	index := ViewSequenceString(view, seq)
	b.prepareLocker.RLock()
	if _, ok := b.prepareSet[index]; ok {
		b.prepareLocker.RUnlock()
		return true
	}
	b.prepareLocker.RUnlock()
	return false
}

func (b *Buffer) FetchPrepareMsg(digest string) (ret *Prepare) {
	ret = nil
	b.prepareLocker.RLock()
	if _, ok := b.prepareBuffer[digest]; !ok {
		log.Printf("[Buffer] error to find prepare msg(%s)", digest[0:9])
		return
	}
	ret = b.prepareBuffer[digest]
	b.prepareLocker.RUnlock()
	return
}

// buffer about precommit
func (b *Buffer) BufferPrecommitMsg(msg *PreCommit) {
	b.preCommitLocker.Lock()
	if _, ok := b.preCommitSet[msg.Digest]; !ok {
		b.preCommitSet[msg.Digest] = make(map[Identify]bool)
	}

	for _, v := range msg.Identifys {
		b.preCommitSet[msg.Digest][v] = true
	}	
	
	b.preCommitLocker.Unlock()
}

func (b *Buffer) ClearPrecommitMsg(digest string) {
	b.preCommitLocker.Lock()
	delete(b.preCommitSet, digest)
	delete(b.preCommitState, digest)
	b.preCommitLocker.Unlock()
}

// buffer about vote
func (b *Buffer) BufferVoteMsg(msg *Vote) {
	b.voteLocker.Lock()
	if _, ok := b.voteSet[msg.Digest]; !ok {
		b.voteSet[msg.Digest] = make(map[Identify]bool)
	}
	b.voteSet[msg.Digest][msg.Identify] = true
	b.voteLocker.Unlock()
}

func (b *Buffer) ClearVoteMsg(digest string) {
	b.voteLocker.Lock()
	delete(b.voteSet, digest)
	delete(b.voteState, digest)
	b.voteLocker.Unlock()
}

func (b *Buffer) ClearVoteAll() {
	b.voteLocker.Lock()
	b.voteSet = make(map[string]map[Identify]bool)
	b.voteState = make(map[string]bool)
	b.voteLocker.Unlock()
}

func (b *Buffer) IsTrueOfVoteMsg(digest string, falut uint) bool {
	b.voteLocker.Lock()
	num := uint(len(b.voteSet[digest]))
	// log.Println("Vote num", num)
	_, ok := b.voteState[digest]
	if num < 2*falut || ok {
		b.voteLocker.Unlock()
		return false
	}
	b.voteState[digest] = true
	b.voteLocker.Unlock()
	return true
}

// buffer about commit
func (b *Buffer) BufferCommitMsg(msg *Commit) {
	b.commitLocker.Lock()
	if _, ok := b.commitSet[msg.Digest]; !ok {
		b.commitSet[msg.Digest] = make(map[Identify]bool)
	}

	for _, v := range msg.Identifys {
		b.commitSet[msg.Digest][v] = true
	}

	b.commitLocker.Unlock()
}

func (b *Buffer) ClearCommitMsg(digest string) {
	b.commitLocker.Lock()
	delete(b.commitSet, digest)
	delete(b.commitState, digest)
	b.commitLocker.Unlock()
}

func (b *Buffer) IsTrueOfCommitMsg(digest string, falut uint) bool {
	b.commitLocker.Lock()
	num := uint(len(b.commitSet[digest]))
	_, ok := b.commitState[digest]
	if num < 2*falut || ok {
		b.commitLocker.Unlock()
		return false
	}
	b.commitState[digest] = true
	b.commitLocker.Unlock()
	return true
}

// buffer about decide
func (b *Buffer) BufferDecideMsg(msg *Decide) {
	b.decideLocker.Lock()
	if _, ok := b.decideSet[msg.Digest]; !ok {
		b.decideSet[msg.Digest] = make(map[Identify]bool)
	}

	for _, v := range msg.Identifys {
		b.decideSet[msg.Digest][v] = true
	}

	b.decideLocker.Unlock()
}

func (b *Buffer) ClearDecideMsg(digest string) {
	b.decideLocker.Lock()
	delete(b.decideSet, digest)
	delete(b.decideState, digest)
	b.decideLocker.Unlock()
}

func (b *Buffer) IsTrueOfDecideMsg(digest string, falut uint) bool {
	b.decideLocker.Lock()
	num := uint(len(b.decideSet[digest]))
	_, ok := b.decideState[digest]
	if num < 2*falut || ok {
		b.decideLocker.Unlock()
		return false
	}
	b.decideState[digest] = true
	b.decideLocker.Unlock()
	return true
}


func (b *Buffer) IsReadyToExecute(digest string, fault uint, view View, sequence Sequence) bool {
	if b.IsExistPrepareMsg(view, sequence) && b.IsTrueOfDecideMsg(digest, fault) {
		return true
	}
	return false
}

// func (b *Buffer) IsReadyToExecute(digest string, fault uint, view View, sequence Sequence) bool {
// 	b.prepareLocker.RLock()
// 	defer b.prepareLocker.RUnlock()

// 	_, isPrepare := b.prepareState[digest]
// 	if b.IsExistPreprepareMsg(view, sequence) && isPrepare && b.IsTrueOfCommitMsg(digest, fault) {
// 		return true
// 	}
// 	return false
// }

// buffer about execute queue, must order by sequence
func (b *Buffer) AppendToExecuteQueue(msg *Prepare) {
	b.executeLocker.Lock()
	// upper bound first index greater than value
	count := len(b.executeQueue)
	first := 0
	for count > 0 {
		step := count / 2
		index := step + first
		if !(msg.Sequence < b.executeQueue[index].Sequence) {
			first = index + 1
			count = count - step - 1
		} else {
			count = step
		}
	}
	// find the first index greater than msg insert into first
	b.executeQueue = append(b.executeQueue, msg)
	copy(b.executeQueue[first+1:], b.executeQueue[first:])
	b.executeQueue[first] = msg
	b.executeLocker.Unlock()
}

func (b *Buffer) BatchExecute(lastSequence Sequence) ([]*Prepare, Sequence) {
	b.executeLocker.Lock()
	batchs := make([]*Prepare, 0)
	index := lastSequence
	loop := 0
	// batch form startSeq sequentially
	for {
		if loop == len(b.executeQueue) {
			b.executeQueue = make([]*Prepare, 0)
			b.executeLocker.Unlock()
			return batchs, index
		}
		if b.executeQueue[loop].Sequence != index+1 {
			b.executeQueue = b.executeQueue[loop:]
			b.executeLocker.Unlock()
			return batchs, index
		}
		batchs = append(batchs, b.executeQueue[loop])
		loop = loop + 1
		index = index + 1
	}
}

// buffer about checkpoint
func (b *Buffer) CheckPoint(sequence Sequence, id Identify) ([]byte, *CheckPoint) {
	clearSet := make(map[Sequence]string, 0)
	minSequence := sequence
	content := ""

	b.prepareLocker.RLock()
	for k, v := range b.prepareBuffer {
		if v.Sequence <= sequence {
			clearSet[v.Sequence] = k
			if v.Sequence < minSequence {
				minSequence = v.Sequence
			}
		}
	}
	b.prepareLocker.RUnlock()

	for minSequence <= sequence {
		d := clearSet[minSequence]
		content = content + d
		minSequence = minSequence + 1
	}

	msg := &CheckPoint{
		Sequence: sequence,
		Digest:   Hash([]byte(content)),
		Id:       id,
	}

	data, err := json.Marshal(msg)
	if err != nil {
		return nil, nil
	}
	return data, msg
}

func (b *Buffer) ClearBuffer(msg *CheckPoint) {
	clearSet := make(map[Sequence]string, 0)
	minSequence := msg.Sequence

	b.prepareLocker.RLock()
	for k, v := range b.prepareBuffer {
		if v.Sequence <= msg.Sequence{
			clearSet[v.Sequence] = k
			if v.Sequence < minSequence {
				minSequence = v.Sequence
			}
		}
	}
	b.prepareLocker.RUnlock()

	for minSequence <= msg.Sequence {
		b.ClearPrepareMsg(clearSet[minSequence])
		b.ClearPrecommitMsg(clearSet[minSequence])
		b.ClearCommitMsg(clearSet[minSequence])
		b.ClearDecideMsg(clearSet[minSequence])
		minSequence = minSequence + 1
	}
}

func (b *Buffer) BufferCheckPointMsg(msg *CheckPoint, id Identify) {
	b.checkPointLocker.Lock()
	if _, ok := b.checkPointBuffer[msg.Digest]; !ok {
		b.checkPointBuffer[msg.Digest] = make(map[Identify]bool)
	}
	b.checkPointBuffer[msg.Digest][id] = true
	b.checkPointLocker.Unlock()
}

func (b *Buffer) IsTrueOfCheckPointMsg(digest string, f uint) (ret bool) {
	ret = false
	b.checkPointLocker.RLock()
	num := uint(len(b.checkPointBuffer[digest]))
	_, ok := b.checkPointState[digest]
	if num < 2*f || ok {
		b.checkPointLocker.RUnlock()
		return
	}
	b.checkPointState[digest] = true
	ret = true
	b.checkPointLocker.RUnlock()
	return
}

func (b *Buffer) Show() {
	log.Printf("[Buffer] node buffer size: prepare(%d) precommit(%d) commit(%d) decide(%d)",
		len(b.prepareBuffer), len(b.prepareSet), len(b.commitSet), len(b.decideSet))
}
