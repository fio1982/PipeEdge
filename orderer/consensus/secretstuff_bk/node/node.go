package node

import (
	"github.com/hyperledger/fabric/orderer/consensus"
	"github.com/hyperledger/fabric/orderer/consensus/secretstuff/cmd"
	"github.com/hyperledger/fabric/orderer/consensus/secretstuff/message"
	"github.com/hyperledger/fabric/orderer/consensus/secretstuff/server"
	"log"
)

var GNode *Node = nil

type Node struct {
	cfg    *cmd.SharedConfig
	server *server.HttpServer

	id       message.Identify
	view     message.View
	table    map[message.Identify]string
	faultNum uint

	primary  message.Identify

	lastReply      *message.LastReply
	sequence       *Sequence
	executeNum     *ExecuteOpNum

	buffer         *message.Buffer

	pkRecv  			chan *message.Election
	startElRecv         chan *message.StartElection
	proveRecv           chan *message.Prove
	requestRecv    		chan *message.Request
	prepareRecv    		chan *message.Prepare
	preCommitRecv  		chan *message.PreCommit
	commitRecv     		chan *message.Commit
	decideRecv     		chan *message.Decide
	prepareVoteRecv     chan *message.Vote
	precommitVoteRecv   chan *message.Vote
	commitVoteRecv      chan *message.Vote
	checkPointRecv 		chan *message.CheckPoint

	prepareSendNotify    	chan bool
	leaderElectionNotify 	chan bool
	leaderElectionNotifyall chan bool
	executeNotify        	chan bool

	// quitPrepare			 chan bool

	supports             map[string]consensus.ConsenterSupport
}

func NewNode(cfg *cmd.SharedConfig, support consensus.ConsenterSupport) *Node {
	node := &Node{
		// config
		cfg:	  cfg,
		// http server
		server:   server.NewServer(cfg),
		// information about node
		id:       cfg.Id,
		view:     cfg.View,
		table:	  cfg.Table,
		faultNum: cfg.FaultNum,

		// primary:  nil,
		// lastReply state
		lastReply:  message.NewLastReply(),
		sequence:   NewSequence(cfg),
		executeNum: NewExecuteOpNum(),
		// the message buffer to store msg
		buffer: message.NewBuffer(),
		// chan for server and recv thread
		pkRecv: 			make(chan *message.Election),
		startElRecv: 		make(chan *message.StartElection),
		proveRecv:          make(chan *message.Prove),
		requestRecv:    	make(chan *message.Request),
		prepareRecv:    	make(chan *message.Prepare),
		preCommitRecv:  	make(chan *message.PreCommit),
		commitRecv:     	make(chan *message.Commit),
		decideRecv:     	make(chan *message.Decide),
		prepareVoteRecv:    make(chan *message.Vote),
		precommitVoteRecv:  make(chan *message.Vote),
		commitVoteRecv:     make(chan *message.Vote),
		checkPointRecv: 	make(chan *message.CheckPoint),
		// chan for notify prepare send thread
		prepareSendNotify: 	make(chan bool),
		leaderElectionNotify: 	 make(chan bool),
		leaderElectionNotifyall: make(chan bool),
		// chan for notify execute op and reply thread
		executeNotify:      make(chan bool, 100),
		// quitPrepare:        make(chan bool),

		supports: 			make(map[string]consensus.ConsenterSupport),
	}
	log.Printf("[Node] the node id:%d, view:%d, fault number:%d\n", node.id, node.view, node.faultNum)
	node.RegisterChain(support)
	return node
}

func (n *Node) RegisterChain(support consensus.ConsenterSupport) {
	if _, ok := n.supports[support.ChainID()]; ok {
		return
	}
	log.Printf("[Node] Register the chain(%s)", support.ChainID())
	n.supports[support.ChainID()] = support
}

// TODO: change threads
func (n *Node) Run() {
	// first register chan for server
	n.server.RegisterChan(n.requestRecv, n.startElRecv, n.pkRecv, n.proveRecv, n.prepareRecv, n.preCommitRecv, n.commitRecv, n.decideRecv, n.prepareVoteRecv, n.precommitVoteRecv, n.commitVoteRecv, n.checkPointRecv)
	go n.server.Run()
	
	go n.leaderElectionThread()
	go n.pkRecvThread()
	go n.proveRecvAndVerifyThread()

	go n.requestRecvThreadSec()

	// go n.requestRecvThread()

	// prepare broadcast
	go n.prepareSendThread()
	// prepare recv and vote
	go n.prepareRecvAndVoteSendThread()

	// vote recv and broadcast precommit
	go n.voteRecvAndPreCommitSendThread()
	// precommit recv and vote
	go n.preCommitRecvAndVoteSendThread()

	// vote recv and broadcast commit
	go n.voteRecvAndCommitSendThread()
	// commit recv and vote
	go n.commitRecvAndVoteSendThread()

	// vote recv and broadcast decide
	go n.voteRecvAndDecideSendThread()
	// decide recv and reply
	go n.decideRecvThread()

	go n.executeAndReplyThread()
	go n.checkPointRecvThread()
}
