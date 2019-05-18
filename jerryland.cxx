#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <tinyalsa/pcm.h>
#include <queue>
#include <thread>
#include <mutex>
#include <cmath>

#include "gpu_fft.h"
#include "mailbox.h"
#include "matlab/matplotlibcpp.h"
namespace plt = matplotlibcpp;


const int fftSize = 12;
const int maxFFTReadSize = 4096;
const int sampleRate = 48000;
const float pi = 3.14159265358979323846;
int peakCount = 1;


// In order to end process, enter sudo kill (the process ID (PID) in the command "top") 
class FrameData
{
	public:
	int frameCount;
	int byteCount;
	void* data;
	
	FrameData(int frameCount, int byteCount)
	{
		data = malloc(byteCount);
		this->frameCount = frameCount;
		this->byteCount = byteCount;
		if (data == NULL) {
			fprintf(stderr, "failed to allocate frames read lol\n");
		}
	}
	
	// Move constructor
	FrameData(FrameData&& theOG)
	{
		frameCount = theOG.frameCount;
		byteCount = theOG.byteCount;
		data = theOG.data;
		theOG.data = nullptr;
	}
	
	FrameData(const FrameData& old_obj) = delete;
	
	~FrameData()
	{
		if (data != nullptr)
			free(data);
	}
};

class PulseCodeModulationBoi
{
	public:
	PulseCodeModulationBoi(int device, int card, int phlagz)
	{
		Open(device, card, phlagz);
	}
	
	~PulseCodeModulationBoi()
	{
		Close();
	}
	
	int Write(const FrameData& fData)
	{
		return pcm_writei(pcm, fData.data, fData.frameCount);
	}
	
	FrameData ReadingRainbow()
	{
		unsigned int frame_size = pcm_frames_to_bytes(pcm, 1);
		unsigned int frames_per_sec = pcm_get_rate(pcm);		
		int read_time = frames_per_sec / 10; // Read 10 milliseconds
		
		FrameData returnedFData(read_time, 
			frame_size * read_time);
			
		int read_count = 
			pcm_readi(pcm, returnedFData.data, read_time);
			
		size_t byte_count = pcm_frames_to_bytes(pcm, read_count);
		
		returnedFData.byteCount = byte_count;
		returnedFData.frameCount = read_count;
		
		return returnedFData;
	}
	
	private:
	struct pcm* pcm;
	void Open(int device, int card, int flags)
	{
		
		struct pcm_config config;
		config.channels = 1;
		config.rate = sampleRate; // Read 48000 points of data per second
		config.format = PCM_FORMAT_S16_LE;
		config.period_size = 1024;
		config.period_count = 2;
		config.start_threshold = 0;
		config.silence_threshold = 0;
		config.stop_threshold = 0;

	    pcm = pcm_open(card, device, flags, &config);
		if (pcm == NULL) {
			fprintf(stderr, "failed to allocate memory for PCM read lol\n");
			return;
		} else if (!pcm_is_ready(pcm)){
			fprintf(stderr, "failed to open PCM read lol\n %s", pcm_get_error(pcm));
			pcm_close(pcm);
			pcm = nullptr;
			return;
		}
	}
	
	void Close()
	{
		if (pcm != nullptr)
		{
			pcm_close(pcm);
		}
		
	}
};

class SFX
{
public:

	bool setPassthrough(bool enable)
	{
		passthrough = enable;
	}

	SFX(std::queue<FrameData>* inputQQ, std::queue<FrameData>* outputQQ,
	 std::mutex* inputMucinex, std::mutex* outputMucinex)
	{
		inputQueue = inputQQ;
		outputQueue = outputQQ;
		inputMutex = inputMucinex;
		outputMutex = outputMucinex;
		mailbox = mbox_open();
		gpu_fft_prepare(mailbox, fftSize, GPU_FFT_FWD, 1, &fftInfo);
		
	}
	~SFX()
	{
		gpu_fft_release(fftInfo);
	}
	
	//void GainThatGrain(FrameData& fData, double grainToGain)
	//{
	//	int16_t* posZero = (int16_t*)fData.data;
	//	for (int i = 0; i < fData.frameCount; i++)
	//	{
	//		posZero[i] = posZero[i] * grainToGain;
	//	}
	//}
	
	//void ConcoctiveOctavesUpOneLmao(FrameData& fData)
	//{
	//	int16_t* posZero = (int16_t*)fData.data;
	//	for (int i = 0; i < fData.frameCount / 2; i++)
	//	{
	//		posZero[i] = posZero[i * 2];
	//	}
	//	fData.byteCount = fData.byteCount / 2;
	//	fData.frameCount = fData.frameCount / 2;
	//}
	
	void BeginTransform()
	{
		while (true)
		{			
			(*inputMutex).lock();
			if (inputQueue->empty())
			{		
				(*inputMutex).unlock();	
			}
			else
			{
				FrameData fData = std::move(inputQueue->front());
				(*inputQueue).pop();
				(*inputMutex).unlock();	
				
				if(!passthrough)
				{
					SineWave(fData);	
				}
				std::lock_guard<std::mutex> lock(*outputMutex);
				outputQueue->push(std::move(fData));
			}

		}
	}
	
private:
	int mailbox;
	struct GPU_FFT* fftInfo;
	std::queue<FrameData>* inputQueue;
	std::queue<FrameData>* outputQueue;
	std::mutex* inputMutex;
	std::mutex* outputMutex;
	bool passthrough;
	
	const double baseNote = 27.5;
	const double halfStepConstant = 1.05946309536;
	
	void SineWave(FrameData& fData) // frequency to output
	{
		std::vector<std::pair<int, float>> frequencies = FastFourierTransform(fData);
		
		
		int16_t* posZero = (int16_t*)fData.data;
		for (int i = 0; i < sampleRate / 10; i++)
		{
			int16_t amplitude = 0;
			for(std::pair<int, float> frequency:frequencies) // <Frequency, Amplitude>
			{				
				float frequencyFactor = frequency.first * (2 * pi / sampleRate);
				amplitude += (int16_t)(std::sin(i * frequencyFactor) * frequency.second / 1000);
			}
			
			posZero[i] = amplitude;
		}
	}
	
	float PythagoreanTheorem(float a, float b)
	{
		return std::sqrt((a * a) + (b * b));
	}
	
	// L1 and L2 average lmao hehexd
	// Everything <= a D5 you can't hear when we bucketize
	// all A notes have no stutter/clicks
	std::vector<std::pair<int, float>> FastFourierTransform(FrameData& fData)
	{
		//std::cout << fData.frameCount << " " << fData.byteCount << " " << fData.data << std::endl;
		
		for (int i = 0; i < maxFFTReadSize; i++)
		{
			struct GPU_FFT_COMPLEX complexBoi;
			complexBoi.re = (float)(((int16_t*)fData.data)[i]);
			complexBoi.im = (float)0;
			(fftInfo->in)[i] = complexBoi;
		}
		gpu_fft_execute(fftInfo);
		
		auto cmp = [](std::pair<int, float> ichi, std::pair<int, float> ni) { return ichi.second > ni.second; };
		std::priority_queue<std::pair<int, float>, std::vector<std::pair<int, float>>, decltype(cmp)> qq(cmp);
		
		int currentBucket = 0;
		double bucketAmplitude = 0;
		int elementsInBucket = 0;
		for (int i = 0; i < maxFFTReadSize / 2; i++)
		{
			float currentAmplitude = PythagoreanTheorem((fftInfo->out)[i].re, (fftInfo->out)[i].im);
			int currentFrequency = (sampleRate * i) / maxFFTReadSize, currentBoi;
			
			if (currentFrequency >= 4186)
				break;
				
			double distance = NumberOfHalfStepsFromANaturalZero(currentFrequency);
			
			if (distance >= currentBucket - 0.5 && distance <= currentBucket + 0.5)
			{
				bucketAmplitude += currentAmplitude;
				elementsInBucket++;
			}
			else
			{
				int finalAmplitude = bucketAmplitude / elementsInBucket;
				//if (currentFrequency <= 133)
				//	finalAmplitude *= 10;
				int finalFrequency = baseNote * std::pow(halfStepConstant, currentBucket);
				std::pair<int, float> currentPair = 
					std::make_pair(finalFrequency, finalAmplitude);
				qq.push(currentPair);
			
				if (qq.size() > peakCount)
					qq.pop();
				
				// std::cout << "Bucket Finalized: " << currentPair.first << "Hz, Amplitude:" << currentPair.second << std::endl;
				
				bucketAmplitude = 0;
				elementsInBucket = 0;
				currentBucket++;
			}
			// std::cout << currentBucket << ", " << elementsInBucket << ", " << bucketAmplitude << std::endl;
		}
		
		std::vector<std::pair<int, float>> data;
		for (int i = 0; i < peakCount; i++)
		{				
			//std::cout << qq.top().first << ", " << qq.top().second << " | ";
			data.push_back(qq.top());
			qq.pop();
		}
		//std::cout << std::endl;
		return data;
	}
	
	double NumberOfHalfStepsFromANaturalZero(double oldFrequency)
	{
		return (std::log(oldFrequency / baseNote)) / (std::log(halfStepConstant));
	}
};

class Input
{
	public:
	Input(std::queue<FrameData>* qq, std::mutex* bm)
	{
		buffer = qq;
		buffer_mutex = bm;
	}
	void BeginRead()
	{
		ReadLimited(-1);
	}
	void ReadLimited(int iterations)
	{
		PulseCodeModulationBoi pcmBoi(0, 1, PCM_IN);
		
		while (iterations == -1 || iterations-- > 0)
		{
			FrameData fData = pcmBoi.ReadingRainbow();
			std::lock_guard<std::mutex> lock(*buffer_mutex);
			//effectsBoi.PrintFFT(fData);
			buffer->push(std::move(fData));
		}
	}
	
	private:
	std::queue<FrameData>* buffer;
	std::mutex* buffer_mutex;
};

class Output
{
	public:
    Output(std::queue<FrameData>* qq, std::mutex* bm)
	{
		buffer = qq;
		buffer_mutex = bm;
	}
	
	void BeginWrite()
	{
		PulseCodeModulationBoi pcmBoi(0, 0, PCM_OUT);
		while (true)
		{
						
			buffer_mutex->lock();	
			if (!buffer->empty())
			{
				FrameData data = std::move(buffer->front());
				buffer->pop();
				buffer_mutex->unlock();
				pcmBoi.Write(data);		
			}
			else
				buffer_mutex->unlock();
		}
	}
	
	private:
	std::queue<FrameData>* buffer;
	std::mutex* buffer_mutex;
};

int main(int argc, char **argv)
{
	const int bufferDelay = 2;
	
	std::mutex outputBuffer_mutex;
	std::mutex inputBuffer_mutex;
	std::queue<FrameData> inputBuffer;
	std::queue<FrameData> outputBuffer;
	Input inputBoi(&inputBuffer, &inputBuffer_mutex);
	Output outputBoi(&outputBuffer, &outputBuffer_mutex);
	SFX sfxBoi(&inputBuffer, &outputBuffer, &inputBuffer_mutex, &outputBuffer_mutex);
	
	inputBoi.ReadLimited(bufferDelay);
	std::thread input(&Input::BeginRead, std::ref(inputBoi));
	std::thread sfx(&SFX::BeginTransform, std::ref(sfxBoi));
	std::thread output(&Output::BeginWrite, std::ref(outputBoi));
	
	
	char choice;
	
	while (true)
	{
		system("clear");
		
		std::cout << "(E)nable FFT" << std::endl << "(D)isable FFT" << 
		std::endl << "(S)ine wave count (currently " << peakCount << ")" 
		<< std::endl << "(Q)uit" << std::endl;
		
		std::cin >> choice;
		
		switch(choice)
		{
			case 'E':
			case 'e':
				sfxBoi.setPassthrough(false);
				break;
			case 'D':
			case 'd':
				sfxBoi.setPassthrough(true);
				break;
			case 'S':
			case 's':
				std::cout << "Enter the number of sine waves to add: ";
				std::cin >> peakCount;
				break;
			case 'Q':
			case 'q':
				return 0;
				break;
			default:
				break;
		}
	}
	
	input.join();
	sfx.join();
	output.join();
	
	
	return 0;
}
