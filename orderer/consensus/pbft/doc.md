## 一、Fabric using  PBFT

* make `configtxgen` can identify `pbft`

```go
// common/tools/configtxgen/localconfig/config.go:388
switch ord.OrdererType {
    case 'pbft':
}
// commom/tools/configtxgen/encoder/encoder.go:38
const ConsensusTypePbft = "pbft"
// commom/tools/configtxgen/encoder/encoder.go:215
switch conf.OrdererType {
	case ConsensusTypePbft:
}
```

* add instance of pbft consensus 

```go
// orderer/common/server/main.go:664
consenters["pbft"] = pbft.New()
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
