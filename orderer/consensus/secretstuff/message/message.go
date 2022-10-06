package message

import (
	"encoding/json"
	cb "github.com/hyperledger/fabric/protos/common"
	"strconv"
)

type TimeStamp uint64 // timestamp format
type Identify uint64  // client identity
type View Identify    // view
type Sequence int64   // sequence

const TYPENORMAL = "normal"
const TYPECONFIG = "config"

// Operation
type Operation struct {
	Envelope  *cb.Envelope
	ChannelID string
	ConfigSeq uint64
	Type      string
}

// Result
type Result struct {
}

// Request
type Request struct {
	Op        Operation `json:"operation"`
	TimeStamp TimeStamp `json:"timestamp"`
	ID        Identify  `json:"clientID"`
}

// Message
type Message struct {
	Requests []*Request `json:"requests"`
}

type Election struct {
	PK		 []byte    `json:"pk"`
	Identify Identify  `json:"id"`
}

type StartElection struct {
	Identify Identify  `json:"id"`
}

type Prove struct {
	PI		 		[]byte    `json:"pi"`
	PK				[]byte    `json:"pk"`
	HASH			[]byte    `json:"hash"`
	Identify 		Identify  `json:"id"`
	// RandomBeacon	[]byte	  `json:"rb"`
}

type Vote struct {
	View     View     `json:"view"`
	Sequence Sequence `json:"sequence"`
	Digest   string   `json:"digest"`
	Identify Identify `json:"id"`
}

// Prepare
type Prepare struct {
	View     View     `json:"view"`
	Sequence Sequence `json:"sequence"`
	Digest   string   `json:"digest"`
	Message  Message  `json:"message"`
}

// Pre-Commit
type PreCommit struct {
	View     View        `json:"view"`
	Sequence Sequence    `json:"sequence"`
	Digest   string      `json:"digest"`
	Identifys []Identify `json:"ids"`
}

// Commit
type Commit struct {
	View     View         `json:"view"`
	Sequence Sequence     `json:"sequence"`
	Digest   string       `json:"digest"`
	Identifys []Identify `json:"ids"`
}

// Decide
type Decide struct {
	View     View        `json:"view"`
	Sequence Sequence    `json:"sequence"`
	Digest   string      `json:"digest"`
	Identifys []Identify `json:"ids"`
}

// Reply
type Reply struct {
	View      View      `json:"view"`
	TimeStamp TimeStamp `json:"timestamp"`
	Id        Identify  `json:"nodeID"`
	Result    Result    `json:"result"`
}

// CheckPoint
type CheckPoint struct {
	Sequence Sequence `json:"sequence"`
	Digest   string	  `json:"digest"`
	Id       Identify `json:"nodeID"`
}

// return byte, msg, digest, error
/** start from leader create prepare message with proof of leader */
func NewPrepareMsg(view View, seq Sequence, batch []*Request) ([]byte, *Prepare, string, error) {
	message := Message{Requests: batch}
	d, err := Digest(message)
	if err != nil {
		return []byte{}, nil, "", nil
	}
	Prepare := &Prepare{
		View:     view,
		Sequence: seq,
		Digest:   d,
		Message:  message,
	}
	ret, err := json.Marshal(Prepare)
	if err != nil {
		return []byte{}, nil, "", nil
	}
	return ret, Prepare, d, nil
}

func NewElectionMsg(msg *Election) ([]byte, *Election, error) {
	election := &Election{
		PK:		  msg.PK,
		Identify: msg.Identify,
	}

	content, err := json.Marshal(election)
	if err != nil {
		return []byte{}, nil, err
	}
	return content, election, nil
}

func NewStartElectionMsg(msg *StartElection) ([]byte, *Election, error) {
	startElection := &Election{
		Identify: msg.Identify,
	}

	content, err := json.Marshal(startElection)
	if err != nil {
		return []byte{}, nil, err
	}
	return content, startElection, nil
}

func NewProveMsg(msg *Prove) ([]byte, *Prove, error) {
	prove := &Prove{
		PI:		  		msg.PI,
		PK:       		msg.PK,
		Identify: 		msg.Identify,
		HASH:           msg.HASH,
		// RandomBeacon: 	msg.RandomBeacon,
	}

	content, err := json.Marshal(prove)
	if err != nil {
		return []byte{}, nil, err
	}
	return content, prove, nil
}

func NewVoteMsg(msg *Vote) ([]byte, *Vote, error) {
	vote := &Vote{
		View:     msg.View,
		Sequence: msg.Sequence,
		Digest:   msg.Digest,
		Identify: msg.Identify,
	}

	content, err := json.Marshal(vote)
	if err != nil {
		return []byte{}, nil, err
	}
	return content, vote, nil
}

// return byte, precommit, error
/** leader collect votes from replicas and create pre-commit message */
func NewPrecommitMsg(msg *PreCommit) ([]byte, *PreCommit, error) {
	precommit := &PreCommit{
		View:      msg.View,
		Sequence:  msg.Sequence,
		Digest:    msg.Digest,
		Identifys: msg.Identifys,
	}
	content, err := json.Marshal(precommit)
	if err != nil {
		return []byte{}, nil, err
	}
	return content, precommit, nil
}

// return byte, commit, error
/** leader collect votes from replicas for confirming precommit and create commit message */
func NewCommitMsg(msg *Commit) ([]byte, *Commit, error) {
	commit := &Commit{
		View:      msg.View,
		Sequence:  msg.Sequence,
		Digest:    msg.Digest,
		Identifys: msg.Identifys,
	}
	content, err := json.Marshal(commit)
	if err != nil {
		return []byte{}, nil, err
	}
	return content, commit, nil
}

// return byte, commit, error
/** leader collect votes from replicas for confirming commit and create decide message */
func NewDecideMsg(msg *Decide) ([]byte, *Decide, error) {
	decide := &Decide{
		View:      msg.View,
		Sequence:  msg.Sequence,
		Digest:    msg.Digest,
		Identifys: msg.Identifys,
	}
	content, err := json.Marshal(decide)
	if err != nil {
		return []byte{}, nil, err
	}
	return content, decide, nil
}

func ViewSequenceString(view View, seq Sequence) string {
	// TODO need better method
	seqStr := strconv.Itoa(int(seq))
	viewStr := strconv.Itoa(int(view))
	seqLen := 4 - len(seqStr)
	viewLen := 28 - len(viewStr)
	// high 4  for viewStr
	for i := 0; i < seqLen; i++ {
		viewStr = "0" + viewStr
	}
	// low  28 for seqStr
	for i := 0; i < viewLen; i++ {
		seqStr = "0" + seqStr
	}
	return viewStr + seqStr
}
