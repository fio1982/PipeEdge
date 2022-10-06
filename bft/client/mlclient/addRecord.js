const MlAccessLedger = require('../mlAccessLedger');

let access = new MlAccessLedger();
var args = process.argv.slice(2)

function addRecord(args) {
    let records = args[0].split(',')
    // console.log(record);
    let items = 7
    for(let i=0;i<records.length;i+=items) {
        let workerId = records[i]
        let publisherId = records[i+1]
        let startTime = records[i+2]
        let endTime = records[i+3]
        let reward = records[i+4]
        let deadline = records[i+5]
        let score = records[i+6]

        // console.log("(workerId, publisherId, startTime, endTime, deadline, score, reward)", workerId, publisherId, startTime, endTime, deadline, score, reward);
        access.createTask(workerId, publisherId, startTime, endTime, deadline, score, reward)
    }
    

    
    // await access.createTask(workerId, publisherId, startTime, endTime, deadline, score, reward)

    // let repus = await access.computeMlReputation(ids)
    // console.log(repus);
    // return repus
}

addRecord(args)


