package node

import (
	"log"
	// "github.com/hyperledger/fabric/orderer/consensus/secretstuff/vrf_ed25519"
	"github.com/hyperledger/fabric/orderer/consensus/secretstuff/vrf"
	"github.com/hyperledger/fabric/orderer/consensus/secretstuff/vrf/sortition"
	"golang.org/x/crypto/ed25519"
	"github.com/hyperledger/fabric/orderer/consensus/secretstuff/server"
	"github.com/hyperledger/fabric/orderer/consensus/secretstuff/message"
	// "crypto/rand"
	// "io"
	// "bytes"
	// "crypto/sha512"
	// "encoding/binary"
	// "encoding/hex"
	
	// "go.dedis.ch/kyber/v3/pairing/bn256"
	// "go.dedis.ch/kyber/v3/sign/bls"
	// "go.dedis.ch/kyber/v3/util/random"

	"net/http"
	"io/ioutil"
	// "reflect"
	"encoding/json"
	"os"
	// "time"
	// "math"
	// "bytes"

	// "context"
	// "fmt"
	// "github.com/drand/drand/client"

	// "github.com/dedis/kyber/pairing/bn256"
	// "github.com/dedis/kyber/sign/bls"
	// "github.com/dedis/kyber/util/random"
	// "encoding/json"
)

func (n *Node) requestRecvThread() {
	log.Printf("[Node] start recv the request thread")
	log.Printf("is primary ? (%t): ", n.IsPrimary())
	for {
		msg := <- n.requestRecv
		// check is primary
		if !n.IsPrimary() {
			if n.lastReply.Equal(msg) {
				// TODO just reply
			}else {
				// TODO just send it to primary
			}
		}
		n.buffer.AppendToRequestQueue(msg)
		n.prepareSendNotify <- true
		// n.leaderElectionNotify <- true
	}
}

// var reqMsg *message.Request
var count message.Sequence
func (n *Node) requestRecvThreadSec() {
	log.Printf("[Node] start recv the request thread")
	n.RemoveFile()
	n.SetSecPrimary(0)
	for {
		msg := <- n.requestRecv
		count = n.GetSEQ()
		count++
		log.Println("-----req: ", count)
		// log.Println("-----n.leaderElectionNotifyall: ", n.leaderElectionNotifyall)
		// log.Println("-----n.leaderElectionNotifyall: ", n.leaderElectionNotify)
		// log.Println("-----n.prepareSendNotify: ", n.prepareSendNotify)
		// start to elect leader
		// log.Printf("[Node] primary: ", n.primary)
		// n.randmonessB()

		n.buffer.AppendToRequestQueue(msg)
		
		// time.Sleep(600 * time.Second)
		if count > 4 {
			// n.leaderElectionNotify <- true
			// n.leaderElectionNotifyall <- true
			log.Println("!!!!start election process!!!!")
			startElection := &message.StartElection{
				Identify: 	n.id,
			}
			content, _, _ := message.NewStartElectionMsg(startElection)
			n.BroadCastAll(content, server.StartElectionEntry)
		} else {
			n.prepareSendNotify <- true
		}
	}
}

var SK []byte
var randomnessBeacon string
func (n *Node) leaderElectionThread() {
	log.Println("[Node] leader election thread")
	for {
		select{
		case msg := <-n.startElRecv:
			resp, err := http.Get("https://api.drand.sh/public/latest")
			if err != nil {
   				log.Fatalln(err)
			}
			type RandomnessB struct {
				Round  				string	`json:round`
				Randomness 			string	`json:randomness`
				Signature			string	`json:signature`
				Previous_signature 	string	`json:previous_signature`
			}
			rb := new(RandomnessB)
			body, _ := ioutil.ReadAll(resp.Body)
			// log.Println("====body====:", string(body))
			json.Unmarshal(body, &rb)

			// log.Println("======= randomness =======", rb.Randomness)
			randomnessBeacon = rb.Randomness
			content, election := n.genKeys()
			// log.Printf("[Election] node(%d) broadcast pk(%d)", election.Identify, election.PK)
			log.Printf("!!!(%s) broadcast keys (%s)", msg.Identify, election)
			n.BroadCastAll(content, server.LeaderElectionEntry)

		// case <-n.leaderElectionNotify:
		// 	log.Println("====leaderElectionNotify====")
		// 	// // gen key
		// 	// pk, sk, _ := ed25519.GenerateKey(nil)
		// 	// // publish public key
		// 	// election := &message.Election{
		// 	// 	PK:			pk,
		// 	// 	Identify: 	n.id,
		// 	// }
		// 	// content, election, _ := message.NewElectionMsg(election)
		// 	content, election := n.genKeys()
		// 	// log.Printf("[Election] node(%d) broadcast pk(%d)", election.Identify, election.PK)
		// 	log.Printf("!!!(%s) broadcast keys", election)
		// 	n.BroadCastAll(content, server.LeaderElectionEntry)
		// 	// log.Println("SK: ", sk)
				
		// case <-n.leaderElectionNotifyall:
		// 	log.Println("====leaderElectionNotify All====")
		// 	content, election := n.genKeys()
		// 	log.Printf("!!!(%s) broadcast keys", election)
		// 	// log.Printf("[Election] node(%d) broadcast pk(%d)", election.Identify, election.PK)
		// 	n.BroadCastAll(content, server.LeaderElectionEntry)
		// 	// log.Println("SK: ", sk)
		}
	}
}

func (n *Node) genKeys() ([]byte, *message.Election) {
	pk, sk, _ := ed25519.GenerateKey(nil)
	SK = sk
	// publish public key
	election := &message.Election{
		PK:			pk,
		Identify: 	n.id,
	}
	content, election, _ := message.NewElectionMsg(election)
	// log.Printf("[Election] node(%d) broadcast pk(%d)", election.Identify, election.PK)
	return content, election
}

func (n *Node) pkRecvThread() {
	pks := make(map[message.Identify][]byte)
	for {
		select {
		case msg := <-n.pkRecv:
			// log.Printf("[Leader Election] recv pk: ", msg)
			pks[msg.Identify] = msg.PK
			// log.Println("[Leader Election] recv pk: ", pks)
			
			// recv enough pk, start to generate prove with own sk
			if uint(len(pks)) >= n.cfg.FaultNum*3+1{
				pi, hash, err := vrf.Prove(pks[n.id], SK, []byte(randomnessBeacon))
				if err != nil {
					log.Printf("Prove error: ", err)
				}
				prove := &message.Prove{
					PI:			  pi,
					PK:           pks[n.id],
					HASH:		  hash,
					Identify: 	  n.id,
					// RandomBeacon: randMsg[:],
				}
				// clear pks
				pks = make(map[message.Identify][]byte)
				content, prove, _ := message.NewProveMsg(prove)
				// broadcast prove
				n.BroadCastAll(content, server.LeaderProveEntry)
			}
		}
	}
}

func (n *Node) proveRecvAndVerifyThread() {
	// var hashs []byte
	// var ratios []float64
	ratios := make(map[message.Identify]float64)
	pks := make(map[message.Identify][]byte) 
	proofs := make(map[message.Identify][]byte) 
	for {
		select{
		case msg := <-n.proveRecv:
			// log.Println("[Node] recv prove: ", msg)
			proverId := msg.Identify
			// proverPK := msg.PK
			// proverPI := msg.PI
			proverH  := msg.HASH

			path, _ := os.Getwd()
			n.GetReputation() 
			
			// log.Println("=== hash ===: ", proverH)
			ratio := sortition.HashRatio(proverH)
			ratios[proverId] = ratio
			pks[proverId] = msg.PK
			proofs[proverId] = msg.PI

			var min float64
			min = 1
			var primaryId message.Identify
			if uint(len(ratios)) > n.cfg.FaultNum*3 {
				log.Println("=== current folder ===: ", path)
				log.Println("=== HashRatio ===: ", ratios)
				for key, value := range ratios {
					if value < min {
						min = value
						primaryId = key
					}
				}
				log.Printf("---primary node (%d) hash value(%s): ", primaryId, min)
				res, err := vrf.Verify(pks[primaryId], proofs[primaryId], []byte(randomnessBeacon))
				
				log.Printf("!!!!!node (%d) Verify result (%s): ", primaryId, res)
				if err != nil {
					log.Println("Verofy error: ", err)
				}
				if res {
					n.SetSecPrimary(primaryId)
				}
				// clear map
				pks = make(map[message.Identify][]byte)
				ratios = make(map[message.Identify]float64)
				proofs = make(map[message.Identify][]byte)

				// check is primary
				if n.IsSecPrimary() {
					log.Printf("[Primary Node] primary: ", n.primary)
					// go n.prepareSendThread()
					n.prepareSendNotify <- true
				}
			}
			

			// n1 := 20
			// n2 := 30
			// n3 := 40
			// n4 := 50
			// log.Println("=== len(proverH ===: ", len(proverH))
			// log.Println("=== 2^len(proverH ===: ", math.Pow(2,float64(len(proverH))))
			// P := proverH/(2^len(proverH))
			// log.Println("=== P ===: ", P)
			// big.Int.Binomial()
			// hashs = append(hashs, proverH)
			// if len(hashs) > 3*n.cfg.FaultNum {
			// 	for _, h := range hashs {
			// 		log.Println("=== hash ===: ", string(h))
			// 	}
			// }

			// for id,pk := range pks {
				
			// 	// verify. if pass, broacast its leadership
			// 	res, err := vrf_ed25519.ECVRF_verify(pk, pi, randMsg[:])
			// 	if err != nil {
			// 		log.Println("Verofy error: ", err)
			// 	}

			// 	log.Println("node (%d) Prove result (%s): ", id, res)
			// }
		}
	}
}

