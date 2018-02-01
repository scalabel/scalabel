package main

import (
	"io/ioutil"
	"os"
	"testing"
)

func TestInit(t *testing.T) {
	expectedDataDir := "../../../data"
	Init(ioutil.Discard, os.Stdout, os.Stdout, os.Stderr)

	if *port != nil || *data_dir != expectedDataDir {
		t.Errorf("got %#v and %#v, wanted %#v and %#v",
			*port, *data_dir, expectedPort, expectedDataDir)
	} else {
		print("Passed TestInit!\n")
	}

}
