package main

import (
	"encoding/json"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
)

var (
	Trace   *log.Logger
	Info    *log.Logger
	Warning *log.Logger
	Error   *log.Logger
	env     Env
)

// Environment details specified in config.json
type Env struct {
	Port    string `json:"port"`
	DataDir string `json:"dataDir"`
}

func Init(
	traceHandle io.Writer,
	infoHandle io.Writer,
	warningHandle io.Writer,
	errorHandle io.Writer) {

	Trace = log.New(traceHandle,
		"TRACE: ",
		log.Ldate|log.Ltime)

	Info = log.New(infoHandle,
		"INFO: ",
		log.Ldate|log.Ltime)

	Warning = log.New(warningHandle,
		"WARNING: ",
		log.Ldate|log.Ltime)

	Error = log.New(errorHandle,
		"ERROR: ",
		log.Ldate|log.Ltime)
}

var HTML []byte
var mux *http.ServeMux

func main() {
	Init(ioutil.Discard, os.Stdout, os.Stdout, os.Stderr)

	// read config file
	cfg, err := ioutil.ReadFile(GetProjPath() + "/config.json")
	if err != nil {
		log.Fatal(err)
	}
	json.Unmarshal(cfg, &env)

	// Mux for static files
	mux = http.NewServeMux()
	mux.Handle("/", http.FileServer(http.Dir(GetProjPath()+"/app")))

	// routes
	http.HandleFunc("/", parse(indexHandler))

	// Simple static handlers can be generated with MakeStandardHandler
	http.HandleFunc("/create",
		MakeStandardHandler("/app/control/create.html"))
	http.HandleFunc("/dashboard",
		MakeStandardHandler("/app/control/monitor.html"))
	http.HandleFunc("/2d_bbox_labeling",
		MakeStandardHandler("/app/annotation/box.html"))
	http.HandleFunc("/2d_road_labeling",
		MakeStandardHandler("/app/annotation/road.html"))
	http.HandleFunc("/2d_seg_labeling",
		MakeStandardHandler("/app/annotation/seg.html"))
	http.HandleFunc("/2d_lane_labeling",
		MakeStandardHandler("/app/annotation/lane.html"))
	http.HandleFunc("/image_labeling",
		MakeStandardHandler("/app/annotation/image.html"))

	http.HandleFunc("/result", readResultHandler)
	http.HandleFunc("/fullResult", readFullResultHandler)

	http.HandleFunc("/postAssignment", postAssignmentHandler)
	http.HandleFunc("/postSubmission", postSubmissionHandler)
	http.HandleFunc("/postLog", postLogHandler)
	http.HandleFunc("/requestAssignment", requestAssignmentHandler)
	http.HandleFunc("/requestSubmission", requestSubmissionHandler)
	http.HandleFunc("/requestInfo", requestInfoHandler)

	http.HandleFunc("/vid_bbox_labeling",
		MakeStandardHandler("/app/annotation/vid.html"))

	log.Fatal(http.ListenAndServe(":"+env.Port, nil))

}
