package main

import (
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"time"

	"gopkg.in/yaml.v2"
)

var (
	// Trace traces error when printed to log
	Trace *log.Logger
	// Info is used for logging non-error/warning info
	Info *log.Logger
	// Warning logs warnings
	Warning *log.Logger
	// Error logs errors
	Error      *log.Logger
	configPath string
)

// Env stores the config info found in config.yml
type Env struct {
	Port           int    `yaml:"port"`
	DataDir        string `yaml:"data"`
	SrcPath        string `yaml:"src"`
	AppSubDir      string `yaml:"appSubDir"`
	Database       string `yaml:"database"`
	ModelGateHost  string `yaml:"modelGateHost"`
	ModelGatePort  string `yaml:"modelGatePort"`
	UserManagement string `yaml:"userManagement"`
	Region         string `yaml:"region"`
	DomainName     string `yaml:"domainName"`
	ClientId       string `yaml:"clientId"`
	RedirectUri    string `yaml:"redirectUri"`
	LogOutUri      string `yaml:"logOutUri"`
	ClientSecret   string `yaml:"clientSecret"`
	AWSTokenUrl    string `yaml:"awsTokenURL"`
	AwsJwkUrl      string `yaml:"awsJwkUrl"`
	UserPoolId     string `yaml:"userPoolID"`
}

func (env Env) AppDir() string {
	return path.Join(env.SrcPath, env.AppSubDir)
}

func (env Env) CreatePath() string {
	return path.Join(env.AppDir(), "control/create.html")
}

func (env Env) DashboardPath() string {
	return path.Join(env.AppDir(), "control/dashboard.html")
}

func (env Env) VendorPath() string {
	return path.Join(env.AppDir(), "control/vendor.html")
}

func (env Env) WorkerPath() string {
	return path.Join(env.AppDir(), "control/worker.html")
}

func (env Env) AdminPath() string {
	return path.Join(env.AppDir(), "control/admin.html")
}

func (env Env) Label2dPath(v string) string {
	if v == "2" {
		return path.Join(env.AppDir(), "annotation/label.html")
	}
	return path.Join(env.AppDir(), "annotation/image.html")
}

func (env Env) Label3dPath(v string) string {
	if v == "2" {
		return path.Join(env.AppDir(), "annotation/label.html")
	}
	return path.Join(env.AppDir(), "annotation/point_cloud.html")
}

func (env Env) PointCloudTrackingPath() string {
	return path.Join(env.AppDir(), "annotation/point_cloud_tracking.html")
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
	cfg, err := ioutil.ReadFile(filepath.Clean(configPath))
	Info.Printf("Configuration:\n%s", cfg)
	if err != nil {
		log.Fatal(err)
	}
	err = yaml.Unmarshal(cfg, &env)
	Info.Println(env)
	if err != nil {
		log.Fatal(err)
	}
	if env.AppSubDir == "" {
		env.AppSubDir = "app/dist"
	}
	return env
}

func InitStorage(database string, dir string) Storage {
	var newStorage Storage
	switch database {
	case "s3":
		newStorage = &S3Storage{}
	case "dynamodb":
		newStorage = &DynamodbStorage{}
	case "local":
		newStorage = &FileStorage{}
	default:
		Error.Panic(fmt.Sprintf("Unknown database %s", database))
	}
	err := newStorage.Init(dir)
	if err != nil {
		Error.Panic(err)
	}
	return newStorage
}

var env Env
var storage Storage
var Users = make(map[string]*User) // store the refreshTokens in the map

func main() {
	Init(ioutil.Discard, os.Stdout, os.Stdout, os.Stderr)

	Error.SetFlags(log.LstdFlags | log.Llongfile)

	env = *NewEnv()
	storage = InitStorage(env.Database, env.DataDir)

	// flow control handlers
	//http.HandleFunc("/", parse(indexHandler))
	http.HandleFunc("/", WrapHandler(http.FileServer(
		http.Dir(path.Join(env.SrcPath, env.AppSubDir)))))
	http.HandleFunc("/dashboard", WrapHandleFunc(dashboardHandler))
	http.HandleFunc("/vendor", WrapHandleFunc(vendorHandler))
	http.HandleFunc("/postProject", WrapHandleFunc(postProjectHandler))
	//http.HandleFunc("/postSatProject", WrapHandleFunc(postSatProjectHandler))
	http.HandleFunc("/postSave", WrapHandleFunc(postSaveHandler))
	http.HandleFunc("/postSaveV2", WrapHandleFunc(postSaveV2Handler))
	http.HandleFunc("/postExport", WrapHandleFunc(postExportHandler))
	http.HandleFunc("/postExportV2", WrapHandleFunc(postExportV2Handler))
	http.HandleFunc("/postDownloadTaskURL",
		WrapHandleFunc(downloadTaskUrlHandler))
	http.HandleFunc("/postLoadAssignment",
		WrapHandleFunc(postLoadAssignmentHandler))
	http.HandleFunc("/postLoadAssignmentV2",
		WrapHandleFunc(postLoadAssignmentV2Handler))
	http.HandleFunc("/auth", authHandler)
	http.HandleFunc("/load", loadHandler)
	http.HandleFunc("/workerDashboard", WrapHandleFunc(workerDashboardHandler))
	http.HandleFunc("/adminDashboard", WrapHandleFunc(adminDashboardHandler))
	http.HandleFunc("/logOut", WrapHandleFunc(logOutHandler))
	http.HandleFunc("/postUsers", WrapHandleFunc(postUsersHandler))
	http.HandleFunc("/postProjectNames",
		WrapHandleFunc(postProjectNamesHandler))
	http.HandleFunc("/postDashboardContents",
		WrapHandleFunc(postDashboardContentsHandler))

	// Simple static handlers can be generated with MakePathHandleFunc
	http.HandleFunc("/create", WrapHandleFunc(createHandler))
	http.HandleFunc("/label2d", WrapHandleFunc(Label2dHandler))
	http.HandleFunc("/label2dv2", WrapHandleFunc(Label2dv2Handler))
	http.HandleFunc("/label3d", WrapHandleFunc(Label3dHandler))
	http.HandleFunc("/label3dv2", WrapHandleFunc(Label3dv2Handler))

	//Get information of the gateway server
	http.HandleFunc("/dev/gateway", WrapHandleFunc(gatewayHandler))

	Info.Printf("Listening to Port %d", env.Port)
	Info.Printf("Local URL: localhost:%d", env.Port)
	server := &http.Server{
		Addr:           fmt.Sprintf(":%d", env.Port),
		Handler:        nil,
		ReadTimeout:    1800 * time.Second,
		WriteTimeout:   1800 * time.Second,
		MaxHeaderBytes: 1 << 20,
	}
	log.Fatal(server.ListenAndServe())
}
