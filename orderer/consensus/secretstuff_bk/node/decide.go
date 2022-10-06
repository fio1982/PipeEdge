package node

import (
	"github.com/hyperledger/fabric/orderer/consensus/secretstuff/message"
	"github.com/hyperledger/fabric/orderer/consensus/secretstuff/server"
	"log"
)

func (n *Node) voteRecvAndDecideSendThread() {
	var voteIds []message.Identify
	for {
		select {
		case msg := <-n.commitVoteRecv:
			log.Printf("[Leader] receive commit vote(%s)(%d) from(%d): ", msg.Digest[0:9], msg.Sequence, msg.Identify)
			n.buffer.BufferVoteMsg(msg)
			voteIds = append(voteIds, msg.Identify)
			if n.buffer.IsTrueOfVoteMsg(msg.Digest, n.cfg.FaultNum) {
				// log.Println("commit voteIds: ", voteIds)
				decide := &message.Decide{
					View:      msg.View,
					Sequence:  msg.Sequence,
					Digest:    msg.Digest,
					Identifys: voteIds,
				}

				voteIds = voteIds[:0]
				log.Printf("[Decide] commit msg(%d) vote success and to broadCast decide", decide.Sequence)
				
				content, decideMsg, err := message.NewDecideMsg(decide)
				if err != nil {
					continue
				}
				n.buffer.BufferDecideMsg(decideMsg)
				// reset vote buffer
				n.buffer.ClearVoteAll()
				n.BroadCastAll(content, server.DecideEntry)
			}
		}
	}
}

// recv decide
func (n *Node) decideRecvThread() {
	for {
		select {
		case msg := <-n.decideRecv:
			log.Printf("[Decide] node recv decide (%d)", msg.Sequence)
			if !n.checkDecideMsg(msg) {
				continue
			}
			// buffer the commit msg
			n.buffer.BufferDecideMsg(msg)

			if n.buffer.IsReadyToExecute(msg.Digest, n.cfg.FaultNum, msg.View, msg.Sequence) {
				n.readytoExecute(msg.Digest)
			}
		}
	}
}

// log.Printf("[Commit] node(%d) vote to the msg(%d)", msg.Identify, msg.Sequence)
// if n.buffer.IsReadyToExecute(msg.Digest, n.cfg.FaultNum, msg.View, msg.Sequence) {
// 	n.readytoExecute(msg.Digest)
// }

func (n *Node) checkDecideMsg(msg *message.Decide) bool {
	if n.view != msg.View {
		return false
	}
	if !n.sequence.CheckBound(msg.Sequence) {
		return false
	}
	return true
}