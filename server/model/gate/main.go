package main

import (
	"flag"
	"fmt"
	"github.com/gorilla/websocket"
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
	configPath string
)

type Configuration struct {
	MachineHost string `yaml:"machineHost"`
	MachinePort string `yaml:"machinePort"`
	Port        int    `yaml:"port"`
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

func main() {
	Init(ioutil.Discard, os.Stdout, os.Stdout, os.Stderr)
	//Read Configuration
	cfg, conf_err := ioutil.ReadFile(configPath)
	if conf_err != nil {
		log.Fatal(conf_err)
	}
	Info.Printf("Configuration:\n%s", cfg)

	configuration := new(Configuration)
	err := yaml.Unmarshal(cfg, &configuration)
	if err != nil {
		log.Fatal(err)
	}

	hub := newhub(configuration)
	go hub.run()
	log.Printf("http server started on port %d\n", configuration.Port)
	http.HandleFunc("/register", func(w http.ResponseWriter, r *http.Request) {
		RegisterServer(hub, w, r)
	})

	err = http.ListenAndServe(fmt.Sprintf(":%d", configuration.Port), nil)
	if err != nil {
		log.Println("Listen and serve error: ", err)
	}
}
func RegisterServer(h *Hub, w http.ResponseWriter, r *http.Request) {
	upg := websocket.Upgrader{ReadBufferSize: 1024,
		WriteBufferSize: 1024,
		CheckOrigin: func(r *http.Request) bool {
			return true
		},
	}
	conn, err := upg.Upgrade(w, r, nil)
	if err != nil {
		log.Println("Register Server Error:", err)
	}
	startSession(h, conn)
}
