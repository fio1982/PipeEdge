package node

// ready to execute the msg(digest) send to execute queue
func (n *Node) readytoExecute(digest string) {
	n.writeFile("end")
	// buffer to ExcuteQueue
	n.buffer.AppendToExecuteQueue(n.buffer.FetchPrepareMsg(digest))
	// notify ExcuteThread
	n.executeNum.Dec()
	// trigger reply
	n.executeNotify<-true
	// n.quitPrepare<-true
}
