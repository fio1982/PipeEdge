/*
 * SPDX-License-Identifier: Apache-2.0
 */

'use strict';

const { Contract } = require('fabric-contract-api');
const { uuid } = require('uuidv4');

class Task extends Contract {

    getRandomInt(min, max) {
        return Math.floor(Math.random() * (max - min + 1) + min);
    }

    async initLedger(ctx) {
        console.info('============= START : Initialize Ledger ===========');
        let numOfServer = 10;
        let tasks = [];
        for (let i = 1; i <= numOfServer; i++) {
            let receiver = 'server' + i
            let task = {
                    taskId: uuid(),
                    executorId: receiver,
                    publisherId: '',
                    startTime: '',
                    endTime: '',
                    deadline: '',
                    baseRewards: String(this.getRandomInt(150, 200)),
                    extraRewards: '0'
            };
            tasks.push(task);
        }

        for (let i = 0; i < tasks.length; i++) {
            await ctx.stub.putState(uuid(), Buffer.from(JSON.stringify(tasks[i])));
            console.info('Added <--> ', tasks[i]);
        }
        console.info('============= END : Initialize Ledger ===========');
    }

    async createTask(ctx, taskId, executorId, publisherId, startTime, endTime, deadline, baseRewards) {
        console.info('============= START : Create Task ===========');
        const P = 0.8;
        startTime = parseFloat(startTime)
        endTime = parseFloat(endTime)
        deadline = parseFloat(deadline)
        baseRewards = parseFloat(baseRewards)

        let actualTime = endTime - startTime;
        let extraRewards = 0;
        if (actualTime < deadline) {
            extraRewards = Math.round(P*((deadline-actualTime) / deadline)*baseRewards, 1);
        } else {
            baseRewards = baseRewards * 0.5;
        }

        const task = {
            "taskId" : taskId,
            "executorId" : executorId,
            "publisherId" : publisherId,
            "startTime" : String(startTime),
            "endTime" : String(endTime),
            "deadline" : String(deadline),
            "baseRewards" : String(baseRewards),
            "extraRewards" : String(extraRewards)
        };

        await ctx.stub.putState(uuid(), Buffer.from(JSON.stringify(task)));
        console.info('============= END : Create Task ===========');
    }

    async queryTasksByExecutor(ctx, queryString) {
        console.info('============= START : Query All Tasks ===========');
        let resultsIterator = await ctx.stub.getQueryResult(queryString);
        
        let results = await this._GetAllResults(resultsIterator);

        console.info('============= END : Query All Tasks ===========');
		return JSON.stringify(results);
    }

    async queryAllTasks(ctx) {
        // all records
        const queryString = `
        {
            "selector": {
               "_id": {
                  "$gt": null
               }
            }
        }`
        console.info('============= START : Query All Tasks ===========');
        let resultsIterator = await ctx.stub.getQueryResult(queryString);
        
        let results = await this._GetAllResults(resultsIterator);

        console.info('============= END : Query All Tasks ===========');
		return JSON.stringify(results);
    }

    async _GetAllResults(iterator) {
		let allResults = [];
		while (true) {
            let res = await iterator.next();
			if (res.value && res.value.value.toString()) {
				let jsonRes = {};
				console.log(res.value.value.toString('utf8'));

				jsonRes.Key = res.value.key;
				try {
					jsonRes.Content = JSON.parse(res.value.value.toString('utf8'));
				} catch (err) {
					console.log(err);
					jsonRes.Content = res.value.value.toString('utf8');
				}
				console.info('============= Get : Query one Tasks ===========', jsonRes);
				allResults.push(jsonRes);
			}
            if (res.done) {
                // explicitly close the iterator
                await iterator.close();
                return allResults;
            }
		}
	}

    /** 
    async queryTask(ctx, carNumber) {
        const carAsBytes = await ctx.stub.getState(carNumber); // get the car from chaincode state
        if (!carAsBytes || carAsBytes.length === 0) {
            throw new Error(`${carNumber} does not exist`);
        }
        console.log(carAsBytes.toString());
        return carAsBytes.toString();
    }

    */
}

module.exports = Task;
