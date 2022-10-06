package node

import (
	"github.com/hyperledger/fabric/orderer/consensus/secretstuff/message"
	"github.com/hyperledger/fabric/orderer/consensus/secretstuff/server"
	"log"
	"time"
)

// send prepare thread by request notify or timer
func (n *Node) prepareSendThread() {
	// TODO change timer duration from config
	// if n.IsSecPrimary() {
		// log.Printf("=== prepareSendThread ==", n.IsSecPrimary())
	log.Printf("=== prepareSendThread ==")
	// n.writeFile("start")
	duration := time.Second
	timer := time.After(duration)
	for {
		select {
		// recv request or time out
		case <-n.prepareSendNotify:
			log.Println("====prepareSendNotify====")
			n.prepareSendHandleFunc()
		case <-timer:
			timer = nil
			n.prepareSendHandleFunc()
			timer = time.After(duration)
		// case <-n.quitPrepare:
		// 	log.Printf("++++ node (%d) quit prepareSendThread ++++", n.id)
		// 	return
		}
	}
	// }
}

var SEQ message.Sequence
/** broadcast proof of leader and proposal*/
func (n *Node) prepareSendHandleFunc() {
	// log.Printf("=== prepareSendHandleFunc ==: ", n.IsSecPrimary())
	// buffer is empty or execute op num max
	n.executeNum.Lock()
	defer n.executeNum.UnLock()
	if n.executeNum.Get() >= n.cfg.ExecuteMaxNum {
		return
	}
	
	if n.buffer.SizeofRequestQueue() < 1 {
		return
	}
	// batch request to discard network traffic
	batch := n.buffer.BatchRequest()
	if len(batch) < 1 {
		return
	}
	// seq + 1
	SEQ++
	seq := SEQ
	log.Println("====SEQ==== ", SEQ)
	// seq = n.sequence.Get()
	n.executeNum.Inc()
	content, msg, digest, err := message.NewPrepareMsg(n.view, seq, batch)
	if err != nil {
		log.Printf("[Prepare] generate prepare message error")
		return
	}
	log.Printf("[Prepare] generate sequence(%d) for msg(%s) request batch size(%d)", seq, digest[0:9], len(batch))
	// buffer the prepare msg
	n.buffer.BufferPrepareMsg(msg)
	// boradcast
	n.BroadCast(content, server.PrepareEntry)
	// TODO send error but buffer the request
}

// non-leader nodes recv prepare and send vote thread
func (n *Node) prepareRecvAndVoteSendThread() {
	// if !n.IsSecPrimary() {
	// log.Printf("Non-leader (%d), IsSecPrimary (%t): ", n.id, n.IsSecPrimary())
	for {
		select {
		case msg := <-n.prepareRecv:
			n.writeFile("start")
			log.Printf("[Prepare] non-leader (%D) view (%d) recv prepare(%d) and send the vote",n.id, msg.View, msg.Sequence)
			if !n.checkPrepareMsg(msg) {
				log.Println("checkPrepareMsg fail continue")
				continue
			}
			// buffer prepare
			n.buffer.BufferPrepareMsg(msg)
			vote := &message.Vote {
				View:     msg.View,
				Sequence: msg.Sequence,
				Digest:   msg.Digest,
				Identify: n.id,
			}
			
			SEQ = msg.Sequence

			// log.Println("Vote: ", vote)
			// send
			content, vote, err := message.NewVoteMsg(vote)
			if err != nil {
				log.Println("create vote error: ", err)
				continue
			}
			// buffer the vote msg, verify 2f backup
			// n.buffer.BufferVoteMsg(vote)
			// boradcast prepare message
			log.Printf("[Prepare] node(%d) vote to the msg(%d)", vote.Identify, vote.Sequence)
			n.SendVote(content, "prepare")
			// when commit and prepare vote success but not recv pre-prepare
			if n.buffer.IsReadyToExecute(msg.Digest, n.cfg.FaultNum, msg.View, msg.Sequence) {
				n.readytoExecute(msg.Digest)
			}
		}
	}
	// }
}

func (n *Node) checkPrepareMsg(msg *message.Prepare) bool {
	// check the same view
	if n.view != msg.View {
		log.Println("same view")
		return false
	}
	// check the same v and n exist diffrent digest
	if n.buffer.IsExistPrepareMsg(msg.View, msg.Sequence) {
		log.Println("same v and n")
		return false
	}
	// check the digest
	d, err := message.Digest(msg.Message)
	log.Printf("message.Digest(msg.Message) (%s): ", d[0:9])
	if err != nil {
		return false
	}
	if d != msg.Digest {
		return false
	}
	// check the n bound
	if !n.sequence.CheckBound(msg.Sequence) {
		return false
	}
	return true
}

func (n *Node) GetSEQ() (message.Sequence) {
	return SEQ
}