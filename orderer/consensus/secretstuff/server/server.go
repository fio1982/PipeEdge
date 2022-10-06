package server

import (
	"github.com/hyperledger/fabric/orderer/consensus/secretstuff/cmd"
	"github.com/hyperledger/fabric/orderer/consensus/secretstuff/message"
	"log"
	"net/http"
	"strconv"
)

/*
	going to be changed to hotstuff phases
*/
const (
	RequestEntry        = "/request"
	LeaderElectionEntry = "/election"
	StartElectionEntry  = "/start"
	LeaderProveEntry    = "/prove"
	PrepareEntry        = "/prepare"
	PreCommitEntry      = "/precommit"
	CommitEntry         = "/commit"
	DecideEntry         = "/decide"
	PrepareVoteEntry    = "/prepareVote"
	PrecommitVoteEntry  = "/precommitVote"
	CommitVoteEntry     = "/commitVote"
	CheckPointEntry     = "/checkpoint"
)

// http listening requests
type HttpServer struct {
	port   int
	server *http.Server

	requestRecv    		chan *message.Request
	startElRecv         chan *message.StartElection
	pkRecv  			chan *message.Election
	proveRecv  			chan *message.Prove
	prepareRecv    		chan *message.Prepare
	preCommitRecv  		chan *message.PreCommit
	commitRecv     		chan *message.Commit
	decideRecv     		chan *message.Decide
	prepareVoteRecv     chan *message.Vote
	precommitVoteRecv   chan *message.Vote
	commitVoteRecv      chan *message.Vote
	checkPointRecv 		chan *message.CheckPoint
}

func NewServer(cfg *cmd.SharedConfig) *HttpServer {
	httpServer := &HttpServer{
		port:   cfg.Port,
		server: nil,
	}
	// set server
	return httpServer
}

// config server: to register the handle chan
func (s *HttpServer) RegisterChan(r chan *message.Request, ser chan *message.StartElection, pk chan *message.Election, pr chan *message.Prove, p chan *message.Prepare, 
	pre chan *message.PreCommit, c chan *message.Commit, d chan *message.Decide, 
	pv chan *message.Vote, pcv chan *message.Vote, cv chan *message.Vote, cp chan *message.CheckPoint) {
	log.Printf("[Server] register the chan for listen func")
	s.requestRecv    	 = r
	s.startElRecv        = ser
	s.pkRecv 			 = pk
	s.proveRecv          = pr
	s.prepareRecv    	 = p
	s.preCommitRecv  	 = pre
	s.commitRecv     	 = c
	s.decideRecv     	 = d
	s.prepareVoteRecv    = pv
	s.precommitVoteRecv  = pcv
	s.commitVoteRecv     = cv
	s.checkPointRecv     = cp
}

// uppercase method is public
func (s *HttpServer) Run() {
	// register server service and run
	log.Printf("[Node] start the listen server")
	s.registerServer()
}

// lowercase method is private
func (s *HttpServer) registerServer() {
	log.Printf("[Server] set listen port:%d\n", s.port)

	httpRegister := map[string]func(http.ResponseWriter, *http.Request){
		RequestEntry:    		s.HttpRequest,
		LeaderElectionEntry:    s.HttpElection,
		StartElectionEntry:     s.HttpStartElection,
		LeaderProveEntry:       s.HttpProve,
		PrepareEntry:    		s.HttpPrepare,
		PreCommitEntry:  		s.HttpPreCommit,
		CommitEntry:     		s.HttpCommit,
		DecideEntry:         	s.HttpDecide,
		PrepareVoteEntry :   	s.HttpPrepareVote,
		PrecommitVoteEntry:		s.HttpPrecommitVote,
		CommitVoteEntry:		s.HttpCommitVote,
		CheckPointEntry: 		s.HttpCheckPoint,
	}

	// log.Println("httpRegister: \n", httpRegister)

	mux := http.NewServeMux()
	for k, v := range httpRegister {
		log.Printf("[Server] register the func for %s", k)
		mux.HandleFunc(k, v)
	}

	// return s
	s.server = &http.Server{
		Addr:    ":" + strconv.Itoa(s.port),
		Handler: mux,
	}

	if err := s.server.ListenAndServe(); err != nil {
		log.Printf("[Server Error] %s", err)
		return
	}
}
