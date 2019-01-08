#ifndef CAFFE_PIXELSHUFFLELAYER_HPP
#define CAFFE_PIXELSHUFFLELAYER_HPP

#include <string>
#include <utility>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

/*Sub-Pixle Layer
  Written by Chao Fan, 2018.07.15
  Parameter:
    upscale_factor_: the scale factor for upsampling;
    output_dim_: the channel of output.
  Notice:
    For GPU, this layer support both forward and backward computation
    But for CPU, this layer just support forward computation
*/
namespace caffe{
    template <typename Dtype>
    class PixelShuffleLayer : public Layer<Dtype>{
        public:
            explicit PixelShuffleLayer(const LayerParameter& param):Layer<Dtype>(param){}
            virtual void LayerSetUp(const vector<Blob<Dtype>*>&bottom, const vector<Blob<Dtype>*>&top);
            virtual void Reshape(const vector<Blob<Dtype>*>&bottom, const vector<Blob<Dtype>*>&top);

            virtual inline const char* type() const { return "PixelShuffleLayer";}
            virtual inline int ExactNumBottomBlobs() const {return 1;}
            virtual inline int ExactNumTopBlobs() const {return 1;}
        
        protected:
            virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
            virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
            
            int upscale_factor_;
            int output_dim_;

            int num_;
            int channels_;
            int height_;
            int width_;
    };
}

#endif
