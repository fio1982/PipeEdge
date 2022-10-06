#!/bin/bash

echo
echo " ____    _____      _      ____    _____ "
echo "/ ___|  |_   _|    / \    |  _ \  |_   _|"
echo "\___ \    | |     / _ \   | |_) |   | |  "
echo " ___) |   | |    / ___ \  |  _ <    | |  "
echo "|____/    |_|   /_/   \_\ |_| \_\   |_|  "
echo
echo "Build your first network (BYFN) end-to-end test"
echo
CHANNEL_NAME="$1"
DELAY="$2"
LANGUAGE="$3"
TIMEOUT="$4"
VERBOSE="$5"
NO_CHAINCODE="$6"
: ${CHANNEL_NAME:="mychannel"}
: ${DELAY:="3"}
: ${LANGUAGE:="node"}
: ${TIMEOUT:="10"}
: ${VERBOSE:="false"}
: ${NO_CHAINCODE:="false"}
LANGUAGE=`echo "$LANGUAGE" | tr [:upper:] [:lower:]`
COUNTER=1
MAX_RETRY=10

CC_SRC_PATH="github.com/chaincode/task/node/"
if [ "$LANGUAGE" = "node" ]; then
	CC_SRC_PATH="/opt/gopath/src/github.com/chaincode/task/node/"
fi

# CC_SRC_PATH="github.com/chaincode/chaincode_example02/go/"
# if [ "$LANGUAGE" = "node" ]; then
# 	CC_SRC_PATH="/opt/gopath/src/github.com/chaincode/chaincode_example02/node/"
# fi

# if [ "$LANGUAGE" = "java" ]; then
# 	CC_SRC_PATH="/opt/gopath/src/github.com/chaincode/chaincode_example02/java/"
# fi

# CC_SRC_PATH="github.com/chaincode/demo"
# if [ "$LANGUAGE" = "node" ]; then
# 	CC_SRC_PATH="/opt/gopath/src/github.com/chaincode/demo"
# fi

# if [ "$LANGUAGE" = "java" ]; then
# 	CC_SRC_PATH="/opt/gopath/src/github.com/chaincode/demo"
# fi


echo "Channel name : "$CHANNEL_NAME

# import utils
. scripts/utils.sh
echo "===================== CORE_PEER_TLS_ENABLED: ${CORE_PEER_TLS_ENABLED}"
createChannel() {
	setGlobals 0 1

	if [ -z "$CORE_PEER_TLS_ENABLED" -o "$CORE_PEER_TLS_ENABLED" = "false" ]; then
                set -x
		peer channel create -o orderer0.example.com:6050 -c $CHANNEL_NAME -f ./channel-artifacts/channel.tx >&log.txt
		res=$?
                set +x
	else
				set -x
		peer channel create -o orderer0.example.com:6050 -c $CHANNEL_NAME -f ./channel-artifacts/channel.tx --tls $CORE_PEER_TLS_ENABLED --cafile $ORDERER_CA >&log.txt
		res=$?
				set +x
	fi

	cat log.txt
	verifyResult $res "Channel creation failed"
	echo "===================== Channel '$CHANNEL_NAME' created ===================== "
	echo
}

joinChannel () {
	# for org in 1 2; do
	#     # for peer in 0 1; do
	# 	joinChannelWithRetry 0 $org
	# 	echo "===================== peer0.org${org} joined channel '$CHANNEL_NAME' ===================== "
	# 	sleep $DELAY
	# 	echo
	#     # done
	# done
	for peer in 0 1; do
		joinChannelWithRetry $peer 1
		echo "===================== peer${peer}.org1 joined channel '$CHANNEL_NAME' ===================== "
		sleep $DELAY
		echo
	done
	# joinChannelWithRetry 2 2
	# 	echo "===================== peer2.org2 joined channel '$CHANNEL_NAME' ===================== "
}

## Create channel
echo "Creating channel..."
createChannel

## Join all the peers to the channel
echo "Having all peers join the channel..."
joinChannel

echo "wait for 10s"
sleep 10

# ## Set the anchor peers for each org in the channel
echo "Updating anchor peers for org1..."
updateAnchorPeers 0 1
# echo "Updating anchor peers for org2..."
# updateAnchorPeers 0 2

if [ "${NO_CHAINCODE}" != "true" ]; then

	## Install chaincode on peer0.org1 and peer0.org2
	echo "Installing chaincode on peer0.org1..."
	installChaincode 0 1

	echo "Installing chaincode on peer1.org1..."
	installChaincode 1 1

	echo "Waiting for Installation request to be committed ..."
	sleep 10

	# Instantiate chaincode on peer0.org1
	# echo "Instantiating chaincode on peer1.org1..."
	# instantiateChaincode 0 1

	# Instantiate chaincode on peer1.org1
	echo "Instantiating chaincode on peer1.org1..."
	instantiateChaincode 1 1



	echo "Waiting for instantiation request to be committed ..."
	sleep 20

	echo "Sending invoke transaction on peer1.org1"
	chaincodeInvoke 1 1

	# # Query chaincode on peer0.org1
	# echo "Querying chaincode on peer0.org1..."
	# chaincodeQuery 0 1 100

	# # Invoke chaincode on peer0.org1 and peer0.org2
	# echo "Sending invoke transaction on peer0.org1 peer0.org2..."
	# chaincodeInvoke 0 1 0 2
	
	# ## Install chaincode on peer1.org2
	# echo "Installing chaincode on peer2.org2..."
	# installChaincode 2 2

	# # Query on chaincode on peer1.org2, check if the result is 90
	# echo "Querying chaincode on peer2.org2..."
	# chaincodeQuery 2 2 90
	
fi


# # 证书文件夹
# PEERROOT=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations
# ORDEROOT=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/ordererOrganizations

# # 节点设置
# ORDERER0NODE=orderer0.example.com:6050
# ORDERER1NODE=orderer1.example.com:6051
# ORDERER2NODE=orderer2.example.com:6052
# ORDERER3NODE=orderer3.example.com:6053

# ORDERERNODE=${ORDERER1NODE}

# PEERORG1NODE=peer0.org1.example.com:7051
# CHANNEL_NAME=mychannel

# NAME=money_demo
# VERSION=1.0

# Org1(){
#     CORE_PEER_MSPCONFIGPATH=${PEERROOT}/org1.example.com/users/Admin@org1.example.com/msp
#     CORE_PEER_ADDRESS=${PEERORG1NODE}
#     CORE_PEER_LOCALMSPID="Org1MSP"
#     echo "node now:peer0.org1.example.com"
# }

# # 安装链码
# InstallChainCode() {
#     Org1
#     peer chaincode install \
#         -n ${NAME} \
#         -v ${VERSION} \
#         -p github.com/chaincode/demo/
#     echo "peer0.org1.example.com install chaincode - demo"
# }

# # 实例链码
# InstantiateChainCode() {
#     peer chaincode instantiate \
#         -o ${ORDERERNODE} \
#         -C ${CHANNEL_NAME} \
#         -n ${NAME} \
#         -v ${VERSION} \
#         -c '{"Args":["Init"]}' \
#         -P "AND ('Org1MSP.peer')"
#     echo "instantiate chaincode"
#     sleep 10
# }

# # 链码测试
# TestDemo() {
#     # 创建账户
#     peer chaincode invoke \
#         -C ${CHANNEL_NAME} \
#         -o ${ORDERERNODE} \
#         -n ${NAME} \
#         --peerAddresses ${PEERORG1NODE} \
#         -c '{"Args":["open","count_a", "100"]}'
#     peer chaincode invoke \
#         -C ${CHANNEL_NAME} \
#         -o ${ORDERERNODE} \
#         -n ${NAME} \
#         --peerAddresses ${PEERORG1NODE} \
#         -c '{"Args":["open","count_b", "100"]}'
#     peer chaincode query \
#         -C ${CHANNEL_NAME} \
#         -n ${NAME} \
#         -c '{"Args":["query","count_a"]}'
#     peer chaincode query \
#         -C ${CHANNEL_NAME} \
#         -n ${NAME} \
#         -c '{"Args":["query","count_b"]}'
#     peer chaincode invoke \
#         -C ${CHANNEL_NAME} \
#         -o ${ORDERERNODE} \
#         -n ${NAME} \
#         --peerAddresses ${PEERORG1NODE} \
#         -c '{"Args":["invoke","count_a","count_b","50"]}'
#     peer chaincode invoke \
#         -C ${CHANNEL_NAME} \
#         -o ${ORDERERNODE} \
#         -n ${NAME} \
#         --peerAddresses ${PEERORG1NODE} \
#         -c '{"Args":["open","count_c", "100"]}'
#     peer chaincode invoke \
#         -C ${CHANNEL_NAME} \
#         -o ${ORDERER3NODE} \
#         -n ${NAME} \
#         --peerAddresses ${PEERORG1NODE} \
#         -c '{"Args":["invoke","count_a","count_c","10"]}'
#     peer chaincode query \
#         -C ${CHANNEL_NAME} \
#         -n ${NAME} \
#         -c '{"Args":["query","count_a"]}'
#     peer chaincode query \
#         -C ${CHANNEL_NAME} \
#         -n ${NAME} \
#         -c '{"Args":["query","count_b"]}'
#     peer chaincode query \
#         -C ${CHANNEL_NAME} \
#         -n ${NAME} \
#         -c '{"Args":["query","count_c"]}'
# }

# InstallChainCode
# InstantiateChainCode
# echo "Waiting for instantiation request to be committed ..."
# sleep 10
# TestDemo

echo
echo "========= All GOOD, BYFN execution completed =========== "
echo

echo
echo " _____   _   _   ____   "
echo "| ____| | \ | | |  _ \  "
echo "|  _|   |  \| | | | | | "
echo "| |___  | |\  | | |_| | "
echo "|_____| |_| \_| |____/  "
echo

exit 0
