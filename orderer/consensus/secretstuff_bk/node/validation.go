package node

import (
    "bufio"
    "log"
    "os"
	"strings"
	"math"
	"strconv"
	"errors"
)

func readFile(filename string) []string {
	f, err := os.Open(filename)
	if err != nil {
        log.Fatal(err)
    }

	scanner := bufio.NewScanner(f)
    var arr []string
    for scanner.Scan() {
		arr = append(arr, scanner.Text())
	}

	return arr
}

func writeFile(filename string, data []string) {
	file, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0755)
 
	if err != nil {
		log.Fatalf("failed creating file: %s", err)
	}
 
	datawriter := bufio.NewWriter(file)
 
	for _, tmp := range data {
		_, _ = datawriter.WriteString(tmp + "\n")
	}
 
	datawriter.Flush()
	file.Close()
}

func Validate(filename string, jobId string, threshold int) (bool, error) {
	tasks := readFile(filename)
	for _, task := range tasks {
		items := strings.Split(task, ",")
		if items[0] == jobId {
			endTime,_ := strconv.Atoi(items[4])
			startTime,_ := strconv.Atoi(items[3])
			deadline,_ := strconv.Atoi(items[5])
			if math.Abs(float64(endTime- startTime - deadline)) <= float64(threshold) {
				return true, nil
			}else {
				return false, nil
			}
		}
	}
	return false, errors.New("can't find job")
}