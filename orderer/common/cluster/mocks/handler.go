// Code generated by mockery v1.0.0. DO NOT EDIT.

package mocks

import mock "github.com/stretchr/testify/mock"
import orderer "github.com/hyperledger/fabric/protos/orderer"

// Handler is an autogenerated mock type for the Handler type
type Handler struct {
	mock.Mock
}

// OnConsensus provides a mock function with given fields: channel, sender, req
func (_m *Handler) OnConsensus(channel string, sender uint64, req *orderer.ConsensusRequest) error {
	ret := _m.Called(channel, sender, req)

	var r0 error
	if rf, ok := ret.Get(0).(func(string, uint64, *orderer.ConsensusRequest) error); ok {
		r0 = rf(channel, sender, req)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// OnSubmit provides a mock function with given fields: channel, sender, req
func (_m *Handler) OnSubmit(channel string, sender uint64, req *orderer.SubmitRequest) error {
	ret := _m.Called(channel, sender, req)

	var r0 error
	if rf, ok := ret.Get(0).(func(string, uint64, *orderer.SubmitRequest) error); ok {
		r0 = rf(channel, sender, req)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}
