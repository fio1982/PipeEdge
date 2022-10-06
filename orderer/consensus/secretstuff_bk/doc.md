## 一、Fabric using  secretstuff

* make `configtxgen` can identify `secretstuff`

```go
// common/tools/configtxgen/localconfig/config.go:388
switch ord.OrdererType {
    case 'secretstuff':
}
// commom/tools/configtxgen/encoder/encoder.go:38
const ConsensusTypeSecretstuff = "secretstuff"
// commom/tools/configtxgen/encoder/encoder.go:215
switch conf.OrdererType {
	case ConsensusTypeSecretstuff:
}
```

* add instance of secretstuff consensus 

```go
// orderer/common/server/main.go:664
// orderer/common/server/main.go:664
Head import add:
"github.com/hyperledger/fabric/orderer/consensus/secretstuff"
consenters["secretstuff"] = secretstuff.New()
```

* implement consensus interface `/orderer/consensus/consensus.go`

```go
// return Chain
type Consenter interface {
	HandleChain(support ConsenterSupport, metadata *cb.Metadata) (Chain, error)
}
// Chain
type Chain interface {
    Order(env *cb.Envelope, configSeq uint64) error
    Configure(config *cb.Envelope, configSeq uint64) error
	WaitReady() error
    Errored() <-chan struct{}
    Start()
    Halt()
}
```

* build orderer image

```
$ make orderer-docker
```

* build configtxgen （output：`.build/bin/configtxgen`）

```
$ make configtxgen
```

## 二、network

| name |       domain       |    IP/port/PBFT port   |   Org   |
| :-------: | :--------------: | :--------------------: | :--------: |
|  Orderer  | orderer0.example.com | 172.22.0.100:6050/6070 | OrdererOrg |
|  Orderer  | orderer1.example.com | 172.22.0.101:6051/6071 | OrdererOrg |
|  Orderer  | orderer2.example.com | 172.22.0.101:6052/6072 | OrdererOrg |
|  Orderer  | orderer3.example.com | 172.22.0.101:6053/6073 | OrdererOrg |
| Peer/Org1 |  peer0.org1.com  |    172.22.0.2:7051     |  Org1MSP   |
| Peer/Org1 |  peer1.org1.com  |    172.22.0.2:8051     |  Org1MSP   |


## implement secret primary
GetPrimary() in node/utils.go

## implement phases
handle.go and server.go in server

## 
cmd.go Getenv("")
PBFT_NODE_ID, PBFT_NODE_TABLE, PBFT_LISTEN_PORT are in docker-compose-base.yaml
change to 
SESTUFF_NODE_ID, SESTUFF_NODE_TABLE, SESTUFF_LISTEN_PORT

## phases messages in message/buffer.go | message.go

## message deliver in node/broadcast.go