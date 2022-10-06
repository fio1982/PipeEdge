const PubNub = require("pubnub");
const prompt = require('prompt-sync')();
const AccessLedger = require('./accessLedger');
const { exec } = require('child_process');
const fs = require('fs');
// const { uuid } = require('uuidv4');
const { v4: uuid } = require('uuid');


ID = ""
candidates = []

const pubnub = new PubNub({
  publishKey: "pub-c-0f0864b8-4a7a-4059-89b9-1d083b503ca6",
  subscribeKey: "sub-c-73b0bad0-500e-11eb-a73a-1eec528e8f1f",
  uuid: "myUniqueUUID",
});

function getRandomInt(min, max) {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min) + min); //The maximum is exclusive and the minimum is inclusive
}

async function publishTask(publisherId, deadline, baseRewards) {
  console.log("publishing task ...");
  let d = new Date();
  let currentTime = d.getTime();
  const result = await pubnub.publish({
    channel: "mychannel",
    message: {
      type: "pub",
      taskId: uuid(),
      publisherId: publisherId,
      deadline: deadline,
      baseRewards: baseRewards,
      publishTime: currentTime.toString()
    },
  });
  // console.log(result);
}

async function sendResToPub(msg) {
  try {
    console.log(`response to publisher: ${JSON.stringify(msg)}`);
    await pubnub.publish({
      channel: "mychannel",
      message: msg
    });
  } catch (error) {
    console.error(`Failed to send response: ${error}`);
  }
}

// function decideExecutor(msg) {
//   console.log(`send decision: ${JSON.stringify(msg)}`);
//   pubnub.publish({
//     channel: "mychannel",
//     message: msg
//   }).then((res) => console.log(res));
 
// }

async function decideExecutor(msg) {
  try {
    console.log(`send decision: ${JSON.stringify(msg)}`);
    await pubnub.publish({
      channel: "mychannel",
      message: msg
    },
    // function(status, response) {
    //   if (status.error) {
    //       console.log("publishing failed w/ status: ", status);
    //   } else {
    //       console.log("message published w/ server response: ", response);
    //   }
    // }
    );
  } catch (error) {
    console.error(`Failed to send response: ${error}`);
  }
}

async function returnDecision(msg) {
  try {
    console.log(`return to decision: ${JSON.stringify(msg)}`);
    await pubnub.publish({
      channel: "mychannel",
      message: msg
    });
  } catch (error) {
    console.error(`Failed to send response: ${error}`);
  }
}

function addListener() {
  pubnub.addListener({
    status: function (statusEvent) {
      // console.log("statusevent: ", statusEvent);
      if (statusEvent.category === "PNConnectedCategory") {
        // publishTask("400", "30");
      }
    },
    message: function (messageEvent) {
      /** candidates get tasks publish, return a response */
      if (ID != messageEvent.message.publisherId && messageEvent.message.type == "pub") {
        console.log("other servers receive publish")
        let taskId = messageEvent.message.taskId;
        let publisherId = messageEvent.message.publisherId;
        let deadline = messageEvent.message.deadline;
        let baseRewards = messageEvent.message.baseRewards;
        let publishTime = messageEvent.message.publishTime;
        
        let msg = {
          type : "res",
          taskId: taskId,
          publisherId: publisherId,
          candidate: ID,
          des : publisherId,
          deadline : deadline,
          baseRewards : baseRewards,
          publishTime: publishTime
        }

        sendResToPub(msg);
      /** publisher gets response from candidates */  
      } else if (messageEvent.message.des == ID && messageEvent.message.type == "res") {
        console.log(messageEvent.message);
        let d = new Date();
        let taskId = messageEvent.message.taskId;
        let currentTime = d.getTime();
        let publisherId = messageEvent.message.publisherId;
        let deadline = messageEvent.message.deadline;
        let baseRewards = messageEvent.message.baseRewards;
        let publishTime = parseFloat(messageEvent.message.publishTime);
        
        let candidate = {}
        candidate['id'] = messageEvent.message.candidate;
        // delay
        candidate['value'] = (currentTime - publishTime).toString();
        candidates.push(candidate);
        // console.log("candidates: ", candidates);

        const maxWaitingTime = 500;
        async function waitFor() {
          let promise = new Promise((res, rej) => {
            setTimeout(() =>  res("timeout, start to choose one!"), maxWaitingTime)
          });
    
          // wait until the promise returns us a value
          let result = await promise;

          console.log(result);
        }

        let msg = {}
         
        // waitFor()
        //   .then(_ =>{
        //     console.log("candidates: ", candidates);
        //     /** 
        //     let access = new AccessLedger();
        //     access.chooseExecutor(candidates)
        //       .then(executor => {
        //         console.log("decide executor: ", executor);
        //         currentTime = d.getTime();
        //         msg = {
        //           type : "decide",
        //           taskId: taskId,
        //           publisherId: publisherId,
        //           executorId: executor,
        //           deadline : deadline,
        //           baseRewards : baseRewards,
        //           // TaskStartTime: currentTime.toString(),
        //           workload: "1000"
        //         };
        //         // reset candidates
        //         candidates = [];
        //         // decideExecutor(msg);
        //       })
        //       .then(() => {
        //         decideExecutor(msg);
        //       });
        //     */
        //   });
          
        if (candidates.length >= 2) {
          // choose executor, call function from accessLedger
          console.log("candidates: ", candidates);
          let access = new AccessLedger();
          access.chooseExecutor(candidates)
            .then(executor => {
              console.log("decide executor: ", executor);
              currentTime = d.getTime();
              msg = {
                type : "decide",
                taskId: taskId,
                publisherId: publisherId,
                executorId: executor,
                deadline : deadline,
                baseRewards : baseRewards,
                // TaskStartTime: currentTime.toString(),
                workload: "1000"
              };
              // reset candidates
              candidates = [];
              // decideExecutor(msg);
            })
            .then(() => {
              decideExecutor(msg);
            });
        }
         
      /** candidates get decision, 
       *  executor start doing task and return results, 
       *  others save task start time*/   
      } else if (messageEvent.message.type == "decide" && messageEvent.message.publisherId != ID) {
        console.log("receive decision: ", messageEvent.message);
        let workload =  parseInt(messageEvent.message.workload);

        let taskId = messageEvent.message.taskId;
        let publisherId = messageEvent.message.publisherId;
        let executorId = messageEvent.message.executorId
        let deadline = messageEvent.message.deadline;
        let baseRewards = messageEvent.message.baseRewards;
        
        if (messageEvent.message.executorId == ID) {
          // do task
          // exec('~/development/ycsb-0.17.0/bin/ycsb.sh run basic -P ~/development/ycsb-0.17.0/workloads/workloada -p operationcount=1000', (err, stdout, stderr) => {
          //   if (err) {
          //     // node couldn't execute the command
          //     return;
          //   }
          //   console.log(stdout);
          //   console.log(stderr);
          // });
          // test doing tasks by wating random time
          setTimeout(() => {
            // return results by publish
            let msg = {
              type: "result",
              taskId: taskId,
              publisherId: publisherId,
              executorId: executorId,
              taskStartTime: new Date().getTime().toString(),
              deadline: deadline,
              baseRewards: baseRewards
            };

            returnDecision(msg);
          }, getRandomInt(400, 600));
          //return
        } else {
          const content = taskId + ',' + executorId + ',' + new Date().getTime();
          console.log("start to write log: ", content);
          // log tart time for validation later
          fs.appendFileSync(__dirname + '/' + ID + '_tasks_time.log', '\n'+content, err => {
            if (err) {
              console.error(err);
              return;
            }
          })
        }
      /** publisher get results and create tx, 
       *  others, except executor, save end time
       * */  
      } else if (messageEvent.message.type == "result" && messageEvent.message.executorId != ID){
        console.log("get results: ", messageEvent.message);
        let taskId = messageEvent.message.taskId;
        let startTime = messageEvent.message.taskStartTime;
        let publisherId = messageEvent.message.publisherId;
        let executorId = messageEvent.message.executorId
        let deadline = messageEvent.message.deadline;
        let baseRewards = messageEvent.message.baseRewards;
        
        // publisher creates transaction
        if (publisherId == ID) {
          console.log("publisher start creating tx....");
          let endTime = new Date().getTime().toString();
          let access = new AccessLedger();
          access.createTask(taskId, executorId, publisherId, startTime, endTime, deadline, baseRewards);
        } else {
          const content =  ',' + new Date().getTime().toString();
          // console.log("content: ", content);
          fs.appendFileSync(__dirname + '/' + ID + '_tasks_time.log', content, err => {
            if (err) {
              console.error(err);
              return;
            }
          })
        }
      }
      // console.log("continuing");
    },
    presence: function (presenceEvent) {
      // handle presence
      console.log("presenceEvent: ", presenceEvent);
    },
  });
}

ID = prompt("input id: ");

addListener()

console.log("Subscribing..");

pubnub.subscribe({
  channels: ["mychannel"],
});

// publishTask("400", "30");

module.exports = {publishTask}
