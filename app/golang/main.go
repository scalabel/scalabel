package main

import (
	"flag"
	"gopkg.in/yaml.v2"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
)

var (
	Trace      *log.Logger
	Info       *log.Logger
	Warning    *log.Logger
	Error      *log.Logger
	configPath = flag.String("config", "", "")
)

// Stores the config info found in config.yml
type Env struct {
	Port        string `yaml:"port"`
	DataDir     string `yaml:"dataDir"`
	ProjectPath string `yaml:"projectPath"`
}

func Init(
	// Initialize all the loggers
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

	// Handle the flags (right now only have config path)
	flag.StringVar(configPath, "s", "", "Path to config.yml")
	flag.Parse()
	if *configPath == "" {
		log.Fatal("Must include --config flag with path to config.yml")
	}
}

// Initialize the environment from the config file
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
	fileServer := http.FileServer(http.Dir(env.DataDir + "/frames"))
	strippedHandler := http.StripPrefix("/frames/", fileServer)
	http.Handle("/frames/", strippedHandler)

	// flow control handlers
	http.HandleFunc("/", parse(indexHandler))
	http.HandleFunc("/dashboard", dashboardHandler)
	http.HandleFunc("/postProject", postProjectHandler)
	http.HandleFunc("/postSave", postSaveHandler)
	http.HandleFunc("/postSubmission", postSubmissionHandler)
	http.HandleFunc("/postLoadTask", postLoadTaskHandler)
	
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

	// labeling handlers
	http.HandleFunc("/2d_bbox_labeling", box2dLabelingHandler)
	http.HandleFunc("/video_bbox_labeling", videoLabelingHandler)

	log.Fatal(http.ListenAndServe(":"+env.Port, nil))
}
