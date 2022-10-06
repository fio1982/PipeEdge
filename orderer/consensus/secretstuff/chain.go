package secretstuff

import (
	"fmt"
	"github.com/hyperledger/fabric/orderer/consensus"
	"github.com/hyperledger/fabric/orderer/consensus/secretstuff/cmd"
	"github.com/hyperledger/fabric/orderer/consensus/secretstuff/message"
	"github.com/hyperledger/fabric/orderer/consensus/secretstuff/node"
	cb "github.com/hyperledger/fabric/protos/common"
	"time"
	"log"
)

type Chain struct {
	exitChan    chan struct{}
	support     consensus.ConsenterSupport
	secretstuffNode	*node.Node
}

func NewChain(support consensus.ConsenterSupport) *Chain {
	// create secretstuff server
	logger.Info("NewChain - ", support.ChainID())
	if node.GNode == nil {
		node.GNode = node.NewNode(cmd.ReadConfig(), support)
		node.GNode.Run()
	} else {
		node.GNode.RegisterChain(support)
	}

	c := &Chain{
		exitChan: make(chan struct{}),
		support:  support,
		secretstuffNode: node.GNode,
	}
	return c
}

func (ch *Chain) Start() {
	logger.Info("start")
}

func (ch *Chain) Errored() <-chan struct{} {
	return ch.exitChan
}

func (ch *Chain) Halt() {
	logger.Info("halt")
	select {
	case <- ch.exitChan:
	default:
		close(ch.exitChan)
	}
}

func (ch *Chain) WaitReady() error {
	logger.Info("wait ready")
	return nil
}

func (ch *Chain) Order(env *cb.Envelope, configSeq uint64) error {
	logger.Info("Normal")
	select {
	case <-ch.exitChan:
		logger.Info("[CHAIN error exit normal]")
		return fmt.Errorf("Exiting")
	default:

	}
	req := &message.Request{
		Op:        message.Operation{
			Envelope:  env,
			ChannelID: ch.support.ChainID(),
			ConfigSeq: configSeq,
			Type:      message.TYPENORMAL,
		},
		TimeStamp: message.TimeStamp(time.Now().UnixNano()),
		ID:        0,
	}

	log.Println("!!!chain Order, send req!!!")
	ch.secretstuffNode.SendSecPrimary(req)
	// ch.secretstuffNode.BroadCastReq(req)
	return nil
}

func (ch *Chain) Configure(config *cb.Envelope, configSeq uint64) error {
	logger.Info("Config")
	select {
	case <-ch.exitChan:
		logger.Info("[CHAIN error exit config]")
		return fmt.Errorf("Exiting")
	default:
	}
	req := &message.Request{
		Op:        message.Operation{
			Envelope:  config,
			ChannelID: ch.support.ChainID(),
			ConfigSeq: configSeq,
			Type:      message.TYPECONFIG,
		},
		TimeStamp: message.TimeStamp(time.Now().UnixNano()),
		ID:        0,
	}

	log.Println("!!!chain Configure, send req!!!")
	ch.secretstuffNode.SendSecPrimary(req)
	// ch.secretstuffNode.BroadCastReq(req)
	return nil
}