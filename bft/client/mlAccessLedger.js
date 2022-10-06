/*
 * SPDX-License-Identifier: Apache-2.0
 */

'use strict';

const { constants } = require('buffer');
const { FileSystemWallet, Gateway } = require('fabric-network');
const path = require('path');
const { uuid } = require('uuidv4');

const ccpPath = path.resolve(__dirname, '..', 'pbft-network-10', 'connection-org1.json');

let contract = null;
class mlAccessLedger { 
    
    async initContract() {
        try {
            let user = 'user3'
            let walletLocalPath = '/home/liang/gopath/src/github.com/hyperledger/fabric/bft/client'
            // console.log("start to init");
            // Create a new file system based wallet for managing identities.
            const walletPath = path.join(walletLocalPath /*process.cwd()*/, 'wallet');
            const wallet = new FileSystemWallet(walletPath);
            // console.log(`Wallet path: ${walletPath}`);
    
            // Check to see if we've already enrolled the user.
            const userExists = await wallet.exists(user);
            if (!userExists) {
                console.log('An identity for the user "user3" does not exist in the wallet');
                console.log('Run the registerUser.js application before retrying');
                return;
            }
    
            // Create a new gateway for connecting to our peer node.
            const gateway = new Gateway();
            await gateway.connect(ccpPath, { wallet, identity: user, discovery: { enabled: true, asLocalhost: true } });
            // console.log("gateway: ", gateway);
            
            // Get the network (channel) our contract is deployed to.
            const network = await gateway.getNetwork('mychannel');
            // console.log("network: ", network);
            // Get the contract from the network.
            const temp = network.getContract('mytask');
            

            contract = temp;   
        } catch (error) {
            console.error(`Failed to evaluate transaction: ${error}`);
            process.exit(1);
        }
    }

    async getTasksByWorker(workerId) {
        try {
            await this.initContract();
            let queryString = {}
            queryString.selector = {};
            queryString.selector.workerId = workerId;
            // queryString.selector.baseRewards = 180;
            // const queryString = `{
            //     "selector": {
            //         "executorId": "server1"
            //     }
            // }`;
            // Evaluate the specified transaction.
            const result = await contract.evaluateTransaction('queryTasksByExecutor', JSON.stringify(queryString));    
            // tasks.push(JSON.parse(result));
            // console.log(JSON.parse(result));
            return JSON.parse(result);
        } catch (error) {
            console.error(`Failed to evaluate transaction: ${error}`);
        }
        
    }

    async getAllTasks() {
        try {
            // console.log("get all tasks");
            await this.initContract();
            // console.log("initContract");
            // Evaluate the specified transaction.
            const result = await contract.evaluateTransaction('queryAllTasks');
            // console.log(`Transaction has been evaluated, result is: ${result.toString()}`);
            return JSON.parse(result);
        } catch (error) {
            console.error(`Failed to evaluate transaction: ${error}`);
        }
    }

    async createTask(workerId, publisherId, startTime, endTime, deadline, score, reward) {
        try {
            await this.initContract();
            let d = new Date();
            let s = d.getTime();
            // Evaluate the specified transaction.
            await contract.submitTransaction('createTaskMl', uuid(), workerId, publisherId, startTime, endTime, deadline, score, reward);
            let end = new Date().getTime();
            console.log(end -s)
            console.log(`Transaction has been submitted`);
        } catch (error) {
            console.error(`Failed to evaluate transaction: ${error}`);
        }
    }

    async computeMlReputation(candidateIds) {
        let candidatesRep = {};
        for (let candidateId of candidateIds) {
            let tasks = await this.getTasksByWorker(candidateId);
            // console.log("tasks: ", tasks);
            let records = []
            tasks.forEach(task => {
                let content = task.Content
                // console.log(content);
                let record = {}
                record = {
                    "worker": content.workerId,
                    "startTime": content.startTime, 
                    "score" : content.score
                }
                records.push(record)
            });

            records = records.filter(record => record.startTime != "");
            let rep = ""
            if (records.length != 0) {
                rep = this.reputationBasedOnScores(records);
                candidatesRep[candidateId] = rep
            } else {
                candidatesRep[candidateId] = 0
            }
            // console.log("rep: ", rep);
        }
        // console.log(candidatesRep);
        return candidatesRep;
    }

    reputationBasedOnScores(taskRecords) {
        const A = 0.4;
        const RECENT_JOB_NUMBER = 20;
        // console.log(taskRecords);
        taskRecords.sort((a, b) => parseInt(b.startTime) - parseInt(a.startTime));
        // let reward_score = {};
        let totalScore = 0;
        // console.log("taskRecords", taskRecords);
        let size = taskRecords.length > RECENT_JOB_NUMBER ? RECENT_JOB_NUMBER : taskRecords.length;
        for (let i=0;i<size;i++) {
            totalScore = A*parseFloat(taskRecords[i].score) + (1-A)*totalScore
        }
        // reward_score['id'] = taskRecords[0]["executor"];
        // reward_score['value'] = score.toString();
        let reward_score = totalScore.toString();
        // console.log(taskRecords);
        // console.log(reward_score);
        return reward_score;

    }
}

module.exports = mlAccessLedger;
