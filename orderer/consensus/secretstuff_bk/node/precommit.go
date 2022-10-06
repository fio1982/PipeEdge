package node

import (
	"github.com/hyperledger/fabric/orderer/consensus/secretstuff/message"
	"github.com/hyperledger/fabric/orderer/consensus/secretstuff/server"
	"log"
)

func (n *Node) voteRecvAndPreCommitSendThread() {
	// log.Println("create voteIds")
	var voteIds []message.Identify
	for {
		select {
		case msg := <-n.prepareVoteRecv:
			// if msg.Type != "prepare" {
			// 	continue
			// }
			n.writeFile("start") 
			log.Printf("[Leader] receive prepare vote(%s)(%d) from(%d): ", msg.Digest[0:9], msg.Sequence, msg.Identify)
			// if !n.checkVoteMsg(msg) {
			// 	continue
			// }
			// buffer the vote msg
			n.buffer.BufferVoteMsg(msg)
			voteIds = append(voteIds, msg.Identify)
			// have enough votes, send precommit msg
			if n.buffer.IsTrueOfVoteMsg(msg.Digest, n.cfg.FaultNum) {
				// log.Println("prepare voteIds: ", voteIds)
				precommit := &message.PreCommit{
					View:      msg.View,
					Sequence:  msg.Sequence,
					Digest:    msg.Digest,
					Identifys: voteIds,
				}
				voteIds = voteIds[:0]
				log.Printf("[Precommit] prepare msg(%d) vote success and to broadCast precommit", precommit.Sequence)
				
				content, precommitMsg, err := message.NewPrecommitMsg(precommit)
				if err != nil {
					continue
				}
				n.buffer.BufferPrecommitMsg(precommitMsg)
				// reset vote buffer
				n.buffer.ClearVoteMsg(precommit.Digest)
				n.BroadCast(content, server.PreCommitEntry)
			}
		}
	}
}

func (n *Node) preCommitRecvAndVoteSendThread() {
	// if !n.IsSecPrimary() {
	for {
		select {
		case msg := <-n.preCommitRecv:
			log.Printf("[Precommit] non-leader recv precommit(%d) and send the vote", msg.Sequence)
			if !n.checkPrecommitMsg(msg) {
				continue
			}
			// buffer the prepare msg
			n.buffer.BufferPrecommitMsg(msg)
			vote := &message.Vote {
				View:     msg.View,
				Sequence: msg.Sequence,
				Digest:   msg.Digest,
				Identify: n.id,
			}
			content, vote, err := message.NewVoteMsg(vote)
			if err != nil {
				continue
			}
			log.Printf("[Precommit] node(%d) vote to the msg(%d)", vote.Identify, vote.Sequence)
			n.SendVote(content, "precommit")
			
			if n.buffer.IsReadyToExecute(msg.Digest, n.cfg.FaultNum, msg.View, msg.Sequence) {
				n.readytoExecute(msg.Digest)
			}
		}
	}
	// }
}

func (n *Node) checkPrecommitMsg(msg *message.PreCommit) bool {
	if n.view != msg.View {
		return false
	}
	if !n.sequence.CheckBound(msg.Sequence) {
		return false
	}
	return true
}

func (n *Node) checkVoteMsg(msg *message.Vote) bool {

	if n.view != msg.View {
		return false
	}
	if !n.sequence.CheckBound(msg.Sequence) {
		return false
	}
	return true
}