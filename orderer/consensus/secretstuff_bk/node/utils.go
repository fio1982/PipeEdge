package node

import (
	"github.com/hyperledger/fabric/orderer/consensus/secretstuff/message"
	"sync"

	"io/ioutil"
    "log"
    "os"
	"strconv"
	"reflect"
	"time"
	// "strings"
)

// the execute op num now in state
type ExecuteOpNum struct {
	num    int
	locker *sync.RWMutex
}

func NewExecuteOpNum() *ExecuteOpNum {
	return &ExecuteOpNum{
		num:    0,
		locker: new(sync.RWMutex),
	}
}

func (n *ExecuteOpNum) Get() int {
	return n.num
}

func (n *ExecuteOpNum) Inc() {
	n.num = n.num + 1
}

func (n *ExecuteOpNum) Dec() {
	n.Lock()
	n.num = n.num - 1
	n.UnLock()
}

func (n *ExecuteOpNum) Lock() {
	n.locker.Lock()
}

func (n *ExecuteOpNum) UnLock() {
	n.locker.Unlock()
}

func (n *Node) GetPrimary() message.Identify {
	all := len(n.table)
	// log.Println("table: ", n.table)
	// log.Println("n.view: ", n.view)
	return message.Identify(int(n.view)%all)
}

/**
 TODO: secret leader
*/
func (n *Node) SetSecPrimary(id message.Identify) {
	log.Println("n.SetSecPrimary(): ", id)
	n.primary = id
}

func (n *Node) GetSecPrimary() message.Identify {
	return n.primary
}

func (n *Node) IsSecPrimary() bool {
	if n.primary == n.id {
		return true
	} else {
		return false
	}
}

func (n *Node) IsPrimary() bool {
	p := n.GetPrimary()
	// log.Println("n.GetPrimary(): ", p)
	// log.Println("message.Identify(n.view)", message.Identify(n.view)) 
	// log.Println("n.id ", n.id ) 
	if p == n.id {
		return true
	}
	return false
}

func (n *Node) RemoveFile() {
	filename := strconv.FormatUint(n.AsUint64(n.id), 10) + "_consensus_time.log"
	err := os.Remove(filename)

	if err != nil {
		log.Printf("error (%s) cannot delete file (%s)", err, filename)
		return
	}
}

func (n *Node) WriteTime(filename string, writeType string) {
	filename = strconv.FormatUint(n.AsUint64(n.id), 10) + "_" + filename
	path, err := os.Getwd()
	if err != nil {
	    log.Println(err)
	}
	f := path+"/"+filename
	log.Printf("start to write (%s) time, filename: (%s) ", writeType, f)
	file, err := os.OpenFile(f, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0755)
	// log.Println("file: ", file)
	if err != nil {
		log.Fatalf("failed creating file: %s", err)
	}
 
	// datawriter := bufio.NewWriter(file)
	now := time.Now().UnixNano() / 1000000
	startTime := strconv.FormatInt(now, 10)
	if writeType == "start" {
		file.WriteString(startTime + ",")
	} else if writeType == "end" {
		file.WriteString(startTime + "\n")
	}
	
 
	file.Sync()
	// file.Close()
}

func (n *Node) writeFile(timeType string) {
	filename := "consensus_time.log"
	n.WriteTime(filename, timeType)
}

func (n *Node) ReadFile(filename string) string {
	path, err := os.Getwd()
	if err != nil {
	    log.Println(err)
	}
	f := path+"/"+filename
	b, err := ioutil.ReadFile(f) 
	if err != nil {
        log.Fatal(err)
    }
	content := string(b)
	log.Println("!!!!file content: ", content)
	/*
	arr := strings.Split(content, "\n")
	for _,str := range arr {
		if len(str) != 0 {
			tmp := strings.Split(str, ",")
			start,_ := strconv.Atoi(tmp[0])
			end, _ := strconv.Atoi(tmp[1])
			log.Println("!!!!consensus time: ", end - start)
		} 
	}
	*/
	return content
	// file, err := os.Open(f)
	// if err != nil {
    //     log.Fatal(err)
    // }

	// scanner := bufio.NewScanner(file)
    // var arr []string
    // for scanner.Scan() {
	// 	arr = append(arr, scanner.Text())
	// }
	// log.Println("!!!!file content: ", arr)
	// return arr
}

func (n *Node) AsUint64(val interface{}) uint64 {
	ref := reflect.ValueOf(val)
	if ref.Kind() != reflect.Uint64 {
		return 0
	}
	return uint64(ref.Uint())
}

func StringCalc(a string, b string) string {
	return ""
}

func (n *Node) GetReputation() string{
	path, err := os.Getwd()
	if err != nil {
	    log.Println(err)
	}
	f := path+"/reputation.log"
	// log.Println("++++ repu file path ++++ ", f)
	b, err := ioutil.ReadFile(f) 
	if err != nil {
        log.Fatal(err)
    }
	content := string(b)

	return content
	// log.Println("!!!!repu file content: ", content)
}
