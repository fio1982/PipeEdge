package node

import (
	"github.com/hyperledger/fabric/orderer/consensus/secretstuff/message"
	"github.com/hyperledger/fabric/orderer/consensus/secretstuff/server"
	"log"
)

func (n *Node) voteRecvAndCommitSendThread() {
	var voteIds []message.Identify
	for {
		select {
		case msg := <-n.precommitVoteRecv:
			log.Printf("[Leader] receive precommit vote(%s)(%d) from(%d): ", msg.Digest[0:9], msg.Sequence, msg.Identify)
			// buffer the precommit vote msg
			n.buffer.BufferVoteMsg(msg)
			voteIds = append(voteIds, msg.Identify)
			if n.buffer.IsTrueOfVoteMsg(msg.Digest, n.cfg.FaultNum) {
				// log.Println("precommit voteIds: ", voteIds)
				commit := &message.Commit{
					View:      msg.View,
					Sequence:  msg.Sequence,
					Digest:    msg.Digest,
					Identifys: voteIds,
				}
				voteIds = voteIds[:0]
				log.Printf("[Commit] precommit msg(%d) vote success and to broadCast commit", commit.Sequence)
				
				content, commitMsg, err := message.NewCommitMsg(commit)
				if err != nil {
					continue
				}
				n.buffer.BufferCommitMsg(commitMsg)
				// reset vote buffer
				n.buffer.ClearVoteAll()
				n.BroadCast(content, server.CommitEntry)
			}
		}
	}
}

func (n *Node) commitRecvAndVoteSendThread() {
	// if !n.IsSecPrimary() {
	for {
		select {
		case msg := <-n.commitRecv:
			log.Printf("[Commit] non-leader recv commit(%d) and send the vote", msg.Sequence)
			if !n.checkCommitMsg(msg) {
				continue
			}
			// buffer the commit msg
			n.buffer.BufferCommitMsg(msg)
			
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
			log.Printf("[Commit] node(%d) vote to the msg(%d)", vote.Identify, vote.Sequence)
			n.SendVote(content, "commit")
			
			if n.buffer.IsReadyToExecute(msg.Digest, n.cfg.FaultNum, msg.View, msg.Sequence) {
				n.readytoExecute(msg.Digest)
			}
		}
	}
	// }
}

func (n *Node) checkCommitMsg(msg *message.Commit) bool {
	if n.view != msg.View {
		return false
	}
	if !n.sequence.CheckBound(msg.Sequence) {
		return false
	}
	return true
}