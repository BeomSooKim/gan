1. GAN 
- batchnorm 사용과, leaky relu 사용시 엄청 큰 효과를 봄(generator)
- noise를 uniform에서 뽑냐, normal에서 뽑냐는 큰 차이를 모르겠음
2. DCGAN
- discriminator에 batchnorm 사용하니 트레이닝 안됨
- discriminator training 시 real과 fake를 합쳐서 하나의 배치로 밀어넣었음
- 이렇게 할때 discriminator에서 batchnorm을 제거하면 어느정도 됨
- discriminator에 batchnorm을 쓴다면, real과 fake를 나눠서 각각 training하면 됨
- 왜인지는 잘 모르겠음..
- vanilla gan에서도 discriminator에 batchnorm을 넣어보니 트레이닝 안됨
- 그리고,,, 코드상오류 중 하나. discriminator training 하고 generator training 할 때 noise vector를 다시 생성해서 함
- 크게 안될건 아닌거같은데.. 논리상 좀 이상한 부분이 있음
