const MlAccessLedger = require('../mlAccessLedger');

let access = new MlAccessLedger();
var args = process.argv.slice(2)

async function getReputations(args) {
    let ids = args[0].split(',')
    // console.log(ids);
    let repus = await access.computeMlReputation(ids)
    console.log(repus);
    // return repus
}

getReputations(args)


