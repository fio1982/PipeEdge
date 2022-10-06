// const { uuid } = require('uuidv4');
const { v4: uuid } = require('uuid');
const AccessLedger = require('./accessLedger')
const mlAccessLedger = require('./mlAccessLedger')

const access = new AccessLedger();
const maccess = new mlAccessLedger();

class Test {
    getRandomInt(min, max) {
        return Math.floor(Math.random() * (max - min + 1) + min);
    }
    async testQueryByExecutor() {
        let executorId = 'server2';
        const result = await access.getTasksByExecutor(executorId);
        // console.log(result);
        console.log(result);
    };

    async getAll() {
        const result = await access.getAllTasks();
        console.log(result);
    }

    testCreate() {
        let d = new Date();
        let n = d.getTime();
        let executorId = "server4"
        let publisherId = "server6"
        let startTime = String(n)
        let endTime = String(n+750)
        let deadline = String(5000)
        let baseRewards = String(15)

        // let taskId = uuid()
        // let epochId = "server4"
        // let price = "750"
        // let buyerId = "5000"
        // let Rewards = "15"
            
        access.createTask(taskId, executorId, publisherId, startTime, endTime, deadline, baseRewards)
        // access.createTask2(epochId, Rewards, buyerId, price)
    }

    testCreateMl() {
        let d = new Date();
        let n = d.getTime();
        let taskId = uuid()
        let workerId = "4"
        let publisherId = "6"
        let startTime = String(n)
        let endTime = String(n+750)
        let deadline = String(5000)
        let score = String(this.getRandomInt(30, 40))
        let reward = String(this.getRandomInt(50, 60))

        // let taskId = uuid()
        // let epochId = "server4"
        // let price = "750"
        // let buyerId = "5000"
        // let Rewards = "15"
            
        maccess.createTask(taskId, workerId, publisherId, startTime, endTime, deadline, score, reward)
        // access.createTask2(epochId, Rewards, buyerId, price)
    }

    testCompRep() {
        let candidates = [{
            id: "server2",
            value : '4444'
        },
        {
            id : "server3",
            value : '3333'
        }
        ]
        
        let executor = access.chooseExecutor(candidates);
        console.log("decide executor: ", executor);
    }

    async testgetMlRep() {
        let candidates = ['2', '4', '5', '7', '8']
        
        let repus = await maccess.computeMlReputation(candidates);
        console.log("decide executor: ", repus);
    }
}


let t = new Test();
t.testgetMlRep()
// t.testCreate()
// for(let i=0;i<5;i++) {
//     t.testCreateMl()
// }



// access.getAllTasks();

// let d = new Date();
// let n = d.getTime();
// let executorId = "server2"
// let publisherId = "server1"
// let startTime = String(n)
// let endTime = String(n+500)
// let deadline = String(600)
// let baseRewards = String(20)

// access.createTask(executorId, publisherId, startTime, endTime, deadline, baseRewards)
