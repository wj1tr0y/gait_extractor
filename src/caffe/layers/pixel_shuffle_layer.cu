//************************************************************************
//Sub-Pixel Layer
//Written by ChaoFan, 2018.07.15
//For GPU, this layer suports both forward and backward compution
//************************************************************************

#include <cfloat>
#include <algorithm>
#include <vector>

#include "caffe/layers/pixel_shuffle_layer.hpp"
#include "caffe/util/gpu_util.cuh"

namespace caffe{
    template <typename Dtype>
    __global__  void PixelShuffleForward(const int nthreads,
        const int upscale_factor, const int output_dim, Dtype* top_data,
        const int num, const int channels, const int height, const int width, const Dtype* bottom_data){
            CUDA_KERNEL_LOOP(index, nthreads){
            
            int bot_channels = channels;
            int bot_height = height;
            int bot_width = width;

            int bot_width_index = index % bot_width;
            int bot_height_index = ((index - bot_width_index) % (bot_width * bot_height)) / bot_width;
            int bot_channel_index = ((index - bot_width_index - bot_height_index * bot_width) % (bot_channels * bot_height * bot_width)) / (bot_height * bot_width);
            int num_index = (index - bot_width_index - bot_height_index * bot_width - bot_channel_index * bot_height * bot_width) / (bot_channels * bot_height * bot_width);

            int top_channel_index = bot_channel_index / (upscale_factor * upscale_factor);
            int top_width_index = bot_width_index * upscale_factor + bot_channel_index % upscale_factor;
            int top_height_index = bot_height_index * upscale_factor + 
                                (bot_channel_index - top_channel_index * upscale_factor * upscale_factor) / upscale_factor;
            int top_data_index = num_index * (output_dim * bot_height * upscale_factor * bot_width * upscale_factor) + 
                                top_channel_index * (bot_height * upscale_factor * bot_width * upscale_factor) + 
                                top_height_index * bot_width * upscale_factor + top_width_index;

            top_data[top_data_index] = (index < num * bot_channels * bot_height * bot_width)?bottom_data[index]:0;
        }
    }
    
    template<typename Dtype>
    void PixelShuffleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
        const Dtype* bottom_data = bottom[0]->gpu_data();
        Dtype* top_data = top[0]->mutable_gpu_data();
        int count = bottom[0]->count();
        caffe_gpu_set(count, Dtype(0), top_data);
        PixelShuffleForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(count, upscale_factor_, output_dim_, top_data, 
                            num_, channels_, height_, width_, bottom_data);
        CUDA_POST_KERNEL_CHECK;
    }

    template <typename Dtype>
    __global__  void PixelShuffleBackward(const int nthreads,
        const int upscale_factor, const int output_dim, const Dtype* top_diff,
        const int num, const int channels, const int height, const int width, Dtype* bottom_diff){
            CUDA_KERNEL_LOOP(index, nthreads){
            
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

    template <typename Dtype>
    void PixelShuffleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, 
                                                const vector<bool>& propagate_down, 
                                                const vector<Blob<Dtype>*>& bottom){
        if(!propagate_down[0]){ return; }
        const Dtype* top_diff = top[0]->gpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        int count = top[0]->count();
        caffe_gpu_set(count, Dtype(0), bottom_diff);
        PixelShuffleBackward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(count, upscale_factor_, output_dim_, top_diff, 
                            num_, channels_, height_, width_, bottom_diff);
        CUDA_POST_KERNEL_CHECK;
    }
    INSTANTIATE_LAYER_GPU_FUNCS(PixelShuffleLayer);
}
