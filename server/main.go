package sat

import (
	"flag"
	"gopkg.in/yaml.v2"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"path"
	"fmt"
)

var (
	Trace      *log.Logger
	Info       *log.Logger
	Warning    *log.Logger
	Error      *log.Logger
	configPath string
)

// Stores the config info found in config.yml
type Env struct {
	Port        int    `yaml:"port"`
	DataDir     string `yaml:"dataDir"`
	ProjectPath string `yaml:"projectPath"`
	AppDir      string `yaml:"AppDir"`
}

func (env Env) CreatePath() string {
	return path.Join(env.ProjectPath, env.AppDir, "control/create.html")
}

func (env Env) MonitorPath() string {
	return path.Join(env.ProjectPath, env.AppDir, "control/monitor.html")
}

func (env Env) VendorPath() string {
	return path.Join(env.ProjectPath, env.AppDir, "control/vendor.html")
}

func (env Env) VideoPath() string {
	return path.Join(env.ProjectPath, env.AppDir, "annotation/video.html")
}

func (env Env) Box2dPath() string {
	return path.Join(env.ProjectPath, env.AppDir, "annotation/box.html")
}

func (env Env) Seg2dPath() string {
	return path.Join(env.ProjectPath, env.AppDir, "annotation/seg.html")
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
	flag.StringVar(&configPath, "config", "", "Path to config.yml")
	flag.Parse()
	if configPath == "" {
		log.Fatal("Must include --config flag with path to config.yml")
	}
}

// Initialize the environment from the config file
func NewEnv() *Env {
	env := new(Env)
	// read config file
	cfg, err := ioutil.ReadFile(configPath)
	Info.Printf("Configuration:\n%s", cfg)
	if err != nil {
		log.Fatal(err)
	}
	err = yaml.Unmarshal(cfg, &env)
	if err != nil {
		log.Fatal(err)
	}
	if env.AppDir == "" {
		env.AppDir = "app/src"
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
	mux.Handle("/", http.FileServer(
		http.Dir(path.Join(env.ProjectPath, env.AppDir))))

	// serve the frames directory
	fileServer := http.FileServer(http.Dir(path.Join(env.DataDir, "/frames")))
	strippedHandler := http.StripPrefix("/frames/", fileServer)
	http.Handle("/frames/", strippedHandler)

	// flow control handlers
	http.HandleFunc("/", parse(indexHandler))
	http.HandleFunc("/dashboard", dashboardHandler)
	http.HandleFunc("/vendor", vendorHandler)
	http.HandleFunc("/postProject", postProjectHandler)
	http.HandleFunc("/postSave", postSaveHandler)
	http.HandleFunc("/postSubmission", postSubmissionHandler)
	http.HandleFunc("/postLoadTask", postLoadTaskHandler)

	// Simple static handlers can be generated with MakeStandardHandler
	http.HandleFunc("/create", MakeStandardHandler(env.CreatePath()))
	http.HandleFunc("/2d_seg_labeling",
		MakeStandardHandler(env.Seg2dPath()))
	//http.HandleFunc("/image_labeling",
	//	MakeStandardHandler(path.Join(appDir, "/annotation/image.html")))

	// labeling handlers
	http.HandleFunc("/2d_bbox_labeling", box2dLabelingHandler)
	http.HandleFunc("/video_bbox_labeling", videoLabelingHandler)

	Info.Printf("Listening to Port %d", env.Port)
	log.Fatal(http.ListenAndServe(fmt.Sprintf(":%d", env.Port), nil))
}
