package secretstuff

import (
	"github.com/hyperledger/fabric/orderer/consensus"
	cb "github.com/hyperledger/fabric/protos/common"
)

type consenter struct{
}

func New() consensus.Consenter {
	return &consenter{}
}

func (secretstuff *consenter) HandleChain(support consensus.ConsenterSupport, metadata *cb.Metadata) (consensus.Chain, error) {
	logger.Info("Handle Chain For secretstuff")
	return NewChain(support), nil
}