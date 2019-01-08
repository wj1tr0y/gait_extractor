/*
  Sub-Pixel Layer
  Written by ChaoFan, 2018.07.15
  For CPU, this layer just supports forward compution
*/

#include <cfloat>
#include <algorithm>

#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/pixel_shuffle_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
    template <typename Dtype>
    void PixelShuffleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
        PixelShuffleParameter pixel_shuffle_param = this->layer_param_.pixel_shuffle_param();

        upscale_factor_ = pixel_shuffle_param.upscale_factor();
        CHECK_GT(upscale_factor_, 1) << "Upsample scale factor must > 1";
        LOG(INFO) << "Upsample scale :" << upscale_factor_;
        
        channels_ = bottom[0]->channels();
        output_dim_ = channels_ / upscale_factor_ / upscale_factor_;
        CHECK_GT(output_dim_, 0) << "Output_dim must > 0";        
        
        
    }

    template <typename Dtype>
    void PixelShuffleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
        num_ = bottom[0]->num();
        height_ = bottom[0]->height();
        width_ = bottom[0]->width();
        top[0]->Reshape(bottom[0]->num(), output_dim_, height_ * upscale_factor_, width_ * upscale_factor_);
    }

    template <typename Dtype>
    static void PixelShuffleForward(const int upscale_factor, const int output_dim, Dtype* top_data, const int count,
                                    const int num, const int channels, const int height, const int width, const Dtype* bottom_data){
        for (int index = 0; index < count; index++){
            
            int width_index = index % width;
            int height_index = ((index - width_index) % (width * height)) / width;
            int channel_index= ((index - width_index - height_index * width) % (channels * height * width)) / (height * width);
            int num_index  = (index - width_index - height_index * width - channel_index * height * width) / (channels * height * width);
            
            int bottom_data_index = index;

            int top_width_index;
            int top_height_index;
            int top_channel_index;
            top_channel_index = channel_index / (upscale_factor * upscale_factor);
            top_width_index = width_index * upscale_factor + channel_index % upscale_factor;
            top_height_index = height_index * upscale_factor + 
                                (channel_index - top_channel_index * upscale_factor * upscale_factor) / upscale_factor;
            int top_data_index = num_index * (output_dim * height * upscale_factor * width * upscale_factor) + 
                                top_channel_index * (height * upscale_factor * width * upscale_factor) + 
                                top_height_index * width * upscale_factor + top_width_index;

            top_data[top_data_index] = bottom_data[bottom_data_index];
        }
    }

    template<typename Dtype>
    void PixelShuffleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* top_data = top[0]->mutable_cpu_data();
        int count = top[0]->count();
        caffe_set(count, Dtype(0), top_data);
        PixelShuffleForward(upscale_factor_, output_dim_, top_data, count,
                            num_, channels_, height_, width_, bottom_data);
    }

    template <typename Dtype>
    static void PixelShuffleBackward(const int upscale_factor, const int output_dim, const Dtype* top_diff, const int count,
                                      const int num, const int channels, const int height, const int width, Dtype* bottom_diff){
        for (int index = 0; index < count; index++){
            
            int top_width = width * upscale_factor;
            int top_height = height * upscale_factor;
            int top_channels = output_dim;

            int top_width_index = index % top_width;
            int top_height_index = ((index - top_width_index) % (top_width * top_height)) / (top_width);
            int top_channel_index = ((index - top_width_index - top_height_index * top_width) % (top_channels * top_height * top_width)) / (top_width * top_height);
            int num_index = (index - top_width_index - top_height_index * top_width - 
                                top_channel_index * (top_height * top_width)) / (top_channels * top_height * top_width);

            int bot_channels = channels;
            int bot_height = height;
            int bot_width = width;

            int bot_channel_index = top_channel_index * upscale_factor * upscale_factor + 
                                (top_width_index % upscale_factor) + (top_height_index % upscale_factor) * upscale_factor; 
            int bot_width_index = top_width_index / upscale_factor;
            int bot_height_index = top_height_index / upscale_factor; 
            int bottom_diff_index = num_index * (bot_channels * bot_height * bot_width) + bot_channel_index * (bot_height * bot_width ) + 
                                    bot_height_index * bot_width + bot_width_index;

            bottom_diff[bottom_diff_index] = (index < num * top_channels * top_height * top_width)?top_diff[index]:0;
        }
    }

    template<typename Dtype>
    void PixelShuffleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& bottom, 
                                                const vector<bool>& propagate_down, 
                                                const vector<Blob<Dtype>*>& top){
        if(!propagate_down[0]){ return; }
        const Dtype* top_diff = top[0]->cpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        int count = top[0]->count();
        caffe_set(count, Dtype(0), bottom_diff);
        PixelShuffleBackward(upscale_factor_, output_dim_, top_diff, count,
                            num_, channels_, height_, width_, bottom_diff);
    }

#ifdef CPU_ONLY
    STUB_GPU(PixelShuffleLayer);
#endif

    INSTANTIATE_CLASS(PixelShuffleLayer);
    REGISTER_LAYER_CLASS(PixelShuffle);
}
