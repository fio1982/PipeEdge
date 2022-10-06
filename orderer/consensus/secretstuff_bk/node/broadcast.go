package node

import (
	"bytes"
	"encoding/json"
	"github.com/hyperledger/fabric/orderer/consensus/secretstuff/message"
	"github.com/hyperledger/fabric/orderer/consensus/secretstuff/server"
	"log"
	"net/http"
)

func (n *Node) SendPrimary(msg *message.Request) {
	content, err := json.Marshal(msg)
	if err != nil {
		log.Printf("error to marshal json")
		return
	}
	// url: /request, call HttpRequest in handle.go, set requestRecv
	go SendPost(content, n.table[n.GetPrimary()] + server.RequestEntry)
}

func (n *Node) SendSecPrimary(msg *message.Request) {
	content, err := json.Marshal(msg)
	if err != nil {
		log.Printf("error to marshal json")
		return
	}
	// url: /request, call HttpRequest in handle.go, set requestRecv
	go SendPost(content,  n.table[n.GetSecPrimary()] + server.RequestEntry)
}

func (n *Node) BroadCastReq(msg *message.Request) {
	content, err := json.Marshal(msg)
	if err != nil {
		log.Printf("error to marshal json")
		return
	}
	for _, v := range n.table {
		go SendPost(content, v + server.RequestEntry)
	}
}

func (n *Node) SendVote(content []byte, voteType string) {
	// url: 'des + /vote', call HttpVote in handle.go, set voteRecv
	entry := ""
	switch voteType {
	case "prepare":
		entry = server.PrepareVoteEntry
	case "precommit":
		entry = server.PrecommitVoteEntry
	case "commit":
		entry = server.CommitVoteEntry
	}
	// go SendPost(content, n.table[n.GetPrimary()] + entry)
	// log.Println("---node---: ", n.table[n.GetSecPrimary()])
	go SendPost(content, n.table[n.GetSecPrimary()] + entry)
}

// // TODO
// func (n *Node) SendVote(msg *message.Vote) {
// 	vote, err := json.Marshal(msg)
// 	if err != nil {
// 		log.Printf("error to marshal json")
// 		return
// 	}
// 	go SendPost(vote, )
// }

func (n *Node) BroadCast(content []byte, handle string) {
	for k, v := range n.table {
		// do not send to my self
		if k == n.id {
			continue
		}
		go SendPost(content, v + handle)
	}
}

func (n *Node) BroadCastAll(content []byte, handle string) {
	for _, v := range n.table {
		go SendPost(content, v + handle)
	}
}

func SendPost(content []byte, url string) {
	buff := bytes.NewBuffer(content)
	if _, err := http.Post(url, "application/json", buff); err != nil {
		log.Printf("[Send] send to %s error: %s", url, err)
	}
}

