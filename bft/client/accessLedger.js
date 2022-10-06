/*
 * SPDX-License-Identifier: Apache-2.0
 */

'use strict';

const { constants } = require('buffer');
const { FileSystemWallet, Gateway } = require('fabric-network');
const path = require('path');

const ccpPath = path.resolve(__dirname, '..', 'pbft-network-10', 'connection-org1.json');

let contract = null;
class AccessLedger { 
    
    async initContract() {
        try {
            let user = 'user3'
            // console.log("start to init");
            // Create a new file system based wallet for managing identities.
            const walletPath = path.join(process.cwd(), 'wallet');
            const wallet = new FileSystemWallet(walletPath);
            console.log(`Wallet path: ${walletPath}`);
    
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

    async getTasksByExecutor(executorId) {
        try {
            await this.initContract();
            let queryString = {}
            queryString.selector = {};
            queryString.selector.executorId = executorId;
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

    async createTask(taskId, executorId, publisherId, startTime, endTime, deadline, baseRewards) {
        try {
            await this.initContract();
            let d = new Date();
            let s = d.getTime();
            // Evaluate the specified transaction.
            await contract.submitTransaction('createTask', taskId, executorId, publisherId, startTime, endTime, deadline, baseRewards);
            let end = new Date().getTime();
            console.log(end -s)
            console.log(`Transaction has been submitted`);
        } catch (error) {
            console.error(`Failed to evaluate transaction: ${error}`);
        }
    }

    async chooseExecutor(candidates) {
        let candidatesIds = []
        candidates.forEach(candidate => candidatesIds.push(candidate['id']));
        // console.log(candidatesIds);
        // let rep = this.computeReputation(candidateIds);
        // console.log(rep);
        let delays = []
        // return reps array
        let reps = await this.computeReputation(candidatesIds);
        let normalizedRep = []
        if (reps.length == 0) {
            candidatesIds.forEach(id => {
                let rep = {"id": id, "value": 0}
                normalizedRep.push(rep);
            })
        } else {
            normalizedRep = this.normalization('rep', reps);
        }
        // console.log("reps: ", reps);
        // normalizedRep = this.normalization('rep', reps);
        // console.log("normalizedRep: ", normalizedRep);
        let normalizedDelay = this.normalization('delay', candidates);
        console.log("normalizedDelay: ", normalizedDelay);
        // console.log(JSON.stringify(reps));

        let executor = this.chooseOne(candidatesIds, normalizedRep, normalizedDelay);
        // console.log("executor: ", executor);
        return executor;
        
    }

    chooseOne(candidatesIds, normalizedRep, normalizedDelay) {
        const repWeight = 0.6;
        const delayWeight = 0.4;

        let scores = [];

        for (let id of candidatesIds) {
            // console.log(id);
            let rep = normalizedRep.filter(nRep => nRep['id'] == id);
            let delay = normalizedDelay.filter(nDelay => nDelay['id'] == id);
            let score = {
                id : id,
                value : parseFloat(rep[0] ? rep[0]['value'] : 0)*repWeight + parseFloat(delay[0] ? delay[0]['value'] : 0)*delayWeight
            }
            // console.log(score);
            scores.push(score);
        }
        scores.sort((a, b) => b.value - a.value);
        return scores[0]['id'];
    }

    async computeReputation(candidateIds) {
        let candidatesRep = [];
        for (let candidateId of candidateIds) {
            let tasks = await this.getTasksByExecutor(candidateId);
            // console.log("tasks: ", tasks);
            let records = []
            tasks.forEach(task => {
                let content = task.Content
                // console.log(content);
                let record = {}
                record = {
                    "executor": content.executorId,
                    "startTime": content.startTime, 
                    "extraRewards" : content.extraRewards
                }
                records.push(record)
            });

            records = records.filter(record => record.startTime != "");
            let rep = ""
            if (records.length != 0) {
                rep = this.reputationBasedOnRewards(records);
                candidatesRep.push(rep);
            } 
            // console.log("rep: ", rep);
        }
        // console.log(candidatesRep);
        return candidatesRep;
    }

    reputationBasedOnRewards(taskRecords) {
        const A = 0.4;
        const RECENT_JOB_NUMBER = 20;
        // console.log(taskRecords);
        taskRecords.sort((a, b) => parseInt(b.startTime) - parseInt(a.startTime));
        let reward_score = {};
        let score = 0;
        // console.log("taskRecords", taskRecords);
        let size = taskRecords.length > RECENT_JOB_NUMBER ? RECENT_JOB_NUMBER : taskRecords.length;
        for (let i=0;i<size;i++) {
            score = A*parseFloat(taskRecords[i].extraRewards) + (1-A)*score
        }
        reward_score['id'] = taskRecords[0]["executor"];
        reward_score['value'] = score.toString();

        // console.log(taskRecords);
        // console.log(reward_score);
        return reward_score;

    }

    normalization(type, data) {
        let sortedData = data.sort((a,b) => parseFloat(a.value) - parseFloat(b.value))
        // console.log(sortedData);
        let max = sortedData[sortedData.length-1]['value'];
        let min = sortedData[0]['value'];

        let normalizedValues = [];
        // console.log(`max: ${max}, min: ${min}`);
        for (let i=0;i<sortedData.length;i++) {
            let normalized = {};
            if (parseFloat(max) === parseFloat(min)) {
                normalized['id'] = sortedData[i]['id'];
                normalized['value'] = 1;
            } else if (type == "rep") {
                // console.log(`sortedData${i}: ${sortedData[i]['id']} : ${sortedData[i]['value']}`);
                normalized['id'] = sortedData[i]['id'];
                normalized['value'] = ((sortedData[i]['value']-min) / (max - min)).toString(); 
            } else if (type == "delay") {
                normalized['id'] = sortedData[i]['id'];
                normalized['value'] = ((max - sortedData[i]['value']) / (max - min)).toString();
            }
            // console.log(`normalized: ${JSON.stringify(normalized)}`);
            normalizedValues.push(normalized);
        }
        // console.log(normalizedValues);
        return normalizedValues;
    }
}

module.exports = AccessLedger;
