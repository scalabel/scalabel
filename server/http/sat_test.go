package main

import (
	"bytes"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path"
	"testing"

	"github.com/aws/aws-sdk-go/aws/session"
)

const (
	ProjectName = "scalabel_test"
	PageTitle   = "TEST PAGE TITLE"
)

func init() {
	Init(ioutil.Discard, os.Stdout, os.Stdout, os.Stderr)
	env = *NewEnv()
}

func addFileToForm(writer *multipart.Writer,
	filePath string, fileField string) error {
	file, err := os.Open(filePath)
	if err != nil {
		return err
	}
	fileContents, err := ioutil.ReadAll(file)
	if err != nil {
		return err
	}
	fi, err := file.Stat()
	if err != nil {
		return err
	}
	file.Close()
	part, err := writer.CreateFormFile(fileField, fi.Name())
	if err != nil {
		return err
	}
	_, err = part.Write(fileContents)
	if err != nil {
		return err
	}
	return nil
}

func TestMain(m *testing.M) {
	_, err := session.NewSession()
	if err == nil {
		storage = &S3Storage{}
		err := storage.Init("us-west-1:scalabel/travis")
		if err != nil {
			Error.Println(err)
		} else {
			m.Run()
		}
		storage = &DynamodbStorage{}
		err = storage.Init("us-west-1")
		if err != nil {
			Error.Println(err)
		} else {
			m.Run()
		}
	}
	storage = &FileStorage{}
	err = storage.Init(env.DataDir)
	if err != nil {
		log.Fatal(err)
	}
	os.Exit(m.Run())
}

func TestPostProject(t *testing.T) {
	body := new(bytes.Buffer)
	writer := multipart.NewWriter(body)
	project, err := GetProject(ProjectName)
	if project.Options.Name != "" {
		t.Fatal(ProjectName + " already exists.")
	}
	writer.WriteField("project_name", ProjectName)
	writer.WriteField("item_type", "image")
	writer.WriteField("label_type", "box2d")
	writer.WriteField("page_title", PageTitle)
	err = addFileToForm(writer, path.Join(env.SrcPath, "examples",
		"image_list.yml"), "item_file")
	if err != nil {
		t.Fatal(err)
	}
	err = addFileToForm(writer, path.Join(env.SrcPath, "examples",
		"categories.yml"), "categories")
	if err != nil {
		t.Fatal(err)
	}
	err = addFileToForm(writer, path.Join(env.SrcPath, "examples",
		"bbox_attributes.yml"), "attributes")
	if err != nil {
		t.Fatal(err)
	}
	writer.WriteField("task_size", "10")
	writer.WriteField("vendor_id", "-1")
	err = writer.Close()
	if err != nil {
		t.Fatal(err)
	}
	req, err := http.NewRequest("POST", "/postProject", body)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Add("Content-Type", writer.FormDataContentType())
	rr := httptest.NewRecorder()
	postProjectHandler(rr, req)
	project, err = GetProject(ProjectName)
	if err != nil {
		t.Fatal(err)
	}
	if project.Options.Name != ProjectName {
		t.Fatal(errors.New("Project name was not saved correctly."))
	}
	tasks, err := GetTasksInProject(ProjectName)
	if err != nil {
		t.Fatal(err)
	}
	if len(tasks) != 10 {
		t.Fatal(errors.New("Incorrect number of tasks in project."))
	}
}

// test dashboard
func TestDashboard(t *testing.T) {
	req, err := http.NewRequest("POST",
		"dashboard?project_name="+ProjectName, nil)
	if err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	dashboardHandler(rr, req)
	if rr.Code != 200 {
		t.Fatal(errors.New(fmt.Sprintf("Dashboard handler HTTP code: %d",
			rr.Code)))
	}
}

// test vendor dashboard
func TestVendorDashboard(t *testing.T) {
	req, err := http.NewRequest("POST", "vendor?project_name="+ProjectName, nil)
	if err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	vendorHandler(rr, req)
	if rr.Code != 200 {
		t.Fatal(errors.New(fmt.Sprintf("Vendor handler HTTP code: %d",
			rr.Code)))
	}
}

func TestLoadAssignment(t *testing.T) {
	for i := 0; i < 10; i += 1 {
		taskJson := `{"task": {"projectOptions": {"name": "%s"}, "index": %d}}`
		buf := bytes.NewBuffer([]byte(fmt.Sprintf(taskJson, ProjectName, i)))
		req, err := http.NewRequest("POST", "postLoadAssignment", buf)
		if err != nil {
			t.Fatal(err)
		}
		rr := httptest.NewRecorder()
		postLoadAssignmentHandler(rr, req)
		if rr.Code != 200 {
			errString := "Load assignment handler HTTP code: %d"
			t.Fatal(errors.New(fmt.Sprintf(errString, rr.Code)))
		}
	}
}

func TestSaveHandler(t *testing.T) {
	for i := 0; i < 10; i += 1 {
		taskJson := `{"task": {"projectOptions": {"name": "%s"}, "index":%d},` +
			`"labels": [{"id": 0, "categoryPath": "test"},` +
			`{"id": 1, "categoryPath": "test"}]}`
		buf := bytes.NewBuffer([]byte(fmt.Sprintf(taskJson, ProjectName, i)))
		req, err := http.NewRequest("POST", "postSave", buf)
		if err != nil {
			t.Fatal(err)
		}
		rr := httptest.NewRecorder()
		postSaveHandler(rr, req)
		if rr.Code != 200 {
			errString := "Save assignment handler HTTP code: %d"
			t.Fatal(errors.New(fmt.Sprintf(errString, rr.Code)))
		}
	}
}

func TestExportHandler(t *testing.T) {
	body := new(bytes.Buffer)
	writer := multipart.NewWriter(body)
	writer.WriteField("project_name", ProjectName)
	err := writer.Close()
	if err != nil {
		t.Fatal(err)
	}
	req, err := http.NewRequest("POST", "/postProject", body)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Add("Content-Type", writer.FormDataContentType())
	rr := httptest.NewRecorder()
	postExportHandler(rr, req)
	if rr.Code != 200 {
		t.Fatal(errors.New(fmt.Sprintf("Export handler HTTP code: %d",
			rr.Code)))
	}
}

func TestDownloadTaskURLHandler(t *testing.T) {
	body := new(bytes.Buffer)
	writer := multipart.NewWriter(body)
	writer.WriteField("project_name", ProjectName)
	err := writer.Close()
	if err != nil {
		t.Fatal(err)
	}
	req, err := http.NewRequest("POST", "postDownloadTaskURL", body)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Add("Content-Type", writer.FormDataContentType())
	rr := httptest.NewRecorder()
	downloadTaskURLHandler(rr, req)
	if rr.Code != 200 {
		errString := "Download task URL handler HTTP code: %d"
		t.Fatal(errors.New(fmt.Sprintf(errString, rr.Code)))
	}
}

func TestDeleteProject(t *testing.T) {
	err := DeleteProject(ProjectName)
	if err != nil {
		t.Fatal(err)
	}
}
