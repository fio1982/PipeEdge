package server

import (
	"encoding/json"
	"github.com/hyperledger/fabric/orderer/consensus/secretstuff/message"
	"log"
	"net/http"
)

func (s *HttpServer) HttpRequest(w http.ResponseWriter, r *http.Request) {
	var msg message.Request
	if err := json.NewDecoder(r.Body).Decode(&msg); err != nil {
		log.Printf("[Http Error] %s", err)
		return
	}
	s.requestRecv <- &msg
}

func (s *HttpServer) HttpElection(w http.ResponseWriter, r *http.Request) {
	var msg message.Election
	if err := json.NewDecoder(r.Body).Decode(&msg); err != nil {
		log.Printf("[Http Error] %s", err)
		return
	}
	s.pkRecv <- &msg
}

func (s *HttpServer) HttpStartElection(w http.ResponseWriter, r *http.Request) {
	var msg message.StartElection
	if err := json.NewDecoder(r.Body).Decode(&msg); err != nil {
		log.Printf("[Http Error] %s", err)
		return
	}
	s.startElRecv <- &msg
}

func (s *HttpServer) HttpProve(w http.ResponseWriter, r *http.Request) {
	var msg message.Prove
	if err := json.NewDecoder(r.Body).Decode(&msg); err != nil {
		log.Printf("[Http Error] %s", err)
		return
	}
	s.proveRecv <- &msg
}

func (s *HttpServer) HttpPrepare(w http.ResponseWriter, r *http.Request) {
	var msg message.Prepare
	if err := json.NewDecoder(r.Body).Decode(&msg); err != nil {
		log.Printf("[Http Error] %s", err)
		return
	}
	s.prepareRecv <- &msg
}

func (s *HttpServer) HttpPreCommit(w http.ResponseWriter, r *http.Request) {
	var msg message.PreCommit
	if err := json.NewDecoder(r.Body).Decode(&msg); err != nil {
		log.Printf("[Http Error] %s", err)
		return
	}
	s.preCommitRecv <- &msg
}

func (s *HttpServer) HttpCommit(w http.ResponseWriter, r *http.Request) {
	var msg message.Commit
	if err := json.NewDecoder(r.Body).Decode(&msg); err != nil {
		log.Printf("[Http Error] %s", err)
		return
	}
	s.commitRecv <- &msg
}

func (s *HttpServer) HttpDecide(w http.ResponseWriter, r *http.Request) {
	var msg message.Decide
	if err := json.NewDecoder(r.Body).Decode(&msg); err != nil {
		log.Printf("[Http Error] %s", err)
		return
	}
	s.decideRecv <- &msg
}

func (s *HttpServer) HttpPrepareVote(w http.ResponseWriter, r *http.Request) {
	var msg message.Vote
	if err := json.NewDecoder(r.Body).Decode(&msg); err != nil {
		log.Printf("[Http Error] %s", err)
		return
	}
	s.prepareVoteRecv <- &msg
}

func (s *HttpServer) HttpPrecommitVote(w http.ResponseWriter, r *http.Request) {
	var msg message.Vote
	if err := json.NewDecoder(r.Body).Decode(&msg); err != nil {
		log.Printf("[Http Error] %s", err)
		return
	}
	s.precommitVoteRecv <- &msg
}

func (s *HttpServer) HttpCommitVote(w http.ResponseWriter, r *http.Request) {
	var msg message.Vote
	if err := json.NewDecoder(r.Body).Decode(&msg); err != nil {
		log.Printf("[Http Error] %s", err)
		return
	}
	s.commitVoteRecv <- &msg
}

func (s *HttpServer) HttpCheckPoint(w http.ResponseWriter, r *http.Request) {
	var msg message.CheckPoint
	if err := json.NewDecoder(r.Body).Decode(&msg); err != nil {
		log.Printf("[Http Error] %s", err)
		return
	}
	s.checkPointRecv <- &msg
}
