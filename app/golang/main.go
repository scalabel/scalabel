package main

import (
	"flag"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"

	"gopkg.in/yaml.v2"
)

var (
	Trace      *log.Logger
	Info       *log.Logger
	Warning    *log.Logger
	Error      *log.Logger
	configPath = flag.String("config", "", "")
)

// Environment details specified in config.json
type Env struct {
	Port        string `yaml:"port"`
	DataDir     string `yaml:"dataDir"`
	ProjectPath string `yaml:"projectPath"`
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

	flag.StringVar(configPath, "s", "", "Path to config.yml")
	flag.Parse()

	if *configPath == "" {
		log.Fatal("Must include --config flag with path to config.yml")
	}
}

func NewEnv() *Env {
	env := new(Env)
	// read config file
	cfg, err := ioutil.ReadFile(*configPath)
	if err != nil {
		log.Fatal(err)
	}
	err = yaml.Unmarshal(cfg, &env)
	if err != nil {
		log.Fatal(err)
	}
	return env
}

var HTML []byte
var mux *http.ServeMux
var env Env

func main() {
	Init(ioutil.Discard, os.Stdout, os.Stdout, os.Stderr)

	env = *NewEnv()

	// Mux for static files
	mux = http.NewServeMux()
	mux.Handle("/", http.FileServer(http.Dir(env.ProjectPath+"/app")))

	// serve the frames directory
	// serveStaticDirectory("data", "frames")
	fileServer := http.FileServer(http.Dir(env.DataDir + "/frames"))
	strippedHandler := http.StripPrefix("/frames/", fileServer)
	http.Handle("/frames/", strippedHandler)

	// routes
	http.HandleFunc("/", parse(indexHandler))

	// Simple static handlers can be generated with MakeStandardHandler
	http.HandleFunc("/create",
		MakeStandardHandler("/app/control/create.html"))
	http.HandleFunc("/2d_road_labeling",
		MakeStandardHandler("/app/annotation/road.html"))
	http.HandleFunc("/2d_seg_labeling",
		MakeStandardHandler("/app/annotation/seg.html"))
	http.HandleFunc("/2d_lane_labeling",
		MakeStandardHandler("/app/annotation/lane.html"))
	http.HandleFunc("/image_labeling",
		MakeStandardHandler("/app/annotation/image.html"))

	http.HandleFunc("/dashboard", dashboardHandler)

	http.HandleFunc("/2d_bbox_labeling", box2DLabelingHandler)

	http.HandleFunc("/result", readResultHandler)
	http.HandleFunc("/fullResult", readFullResultHandler)

	http.HandleFunc("/postAssignment", postAssignmentHandler)
	http.HandleFunc("/postSubmission", postSubmissionHandler)
	http.HandleFunc("/postLog", postLogHandler)
	http.HandleFunc("/requestAssignment", requestAssignmentHandler)
	http.HandleFunc("/requestSubmission", requestSubmissionHandler)
	http.HandleFunc("/requestInfo", requestInfoHandler)

	http.HandleFunc("/postVideoAssignment", postVideoAssignmentHandler)
	http.HandleFunc("/video_bbox_labeling", videoLabelingHandler)

	log.Fatal(http.ListenAndServe(":"+env.Port, nil))

}
