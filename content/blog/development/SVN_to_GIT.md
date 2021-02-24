---
title: 'SVN to Git'
date: 2021-02-03 14:43:13
category: 'development'
draft: false
---

git 2.27.0 windows.1 에서는 에러 발생 git svn command 에러

2.26으로 다운 혹은 3.0으로 업데이트

git 2.X 버전 사용 단, 2.27 제외

docker centos7 python3.6 설치

yum keyboardinterrupt, e

## SVN 사용자 목록 가져오기

```bash
svn log -q | awk -F '|' '/^r/ {sub("^ ", "", $2); sub(" $", "", $2); print $2" = "$2" <"$2">"}' | sort -u > authors.txt
```

출력 결과 authors.txt 파일이 생성된다.

해당 파일에 이메일을 추가하여 매핑이 가능하도록 수정한다. 만일 해당 작성자가 현재 없는 사람이라면 임의로 이메일 주소를 입력한다.

```xml
author1 = author1 <author1>
author2 = author2 <author2>
author3 = author2 <author3>
```

```xml
author1 = author1 <이메일주소>
author2 = author2 <이메일주소>
author3 = author2 <이메일주소>
```

## SVN clone using Git command

```bash
git svn clone [SVN Repository URL] --trunk=trunk --branches=branches --tags=tags --authors-file=authors.txt --no-metadata -s [출력 폴더명]
```

## Local ⇒ Remote (원격)

clone 한 폴더로 이동

원격 repository에 연동

이때 업로드할 소스의 용량이 큰 경우 SSH 주소를 사용하는 것이 좋다.

[git SSH Key 설정법](https://www.notion.so/git-SSH-Key-96b8490c11f947c4864d3fa3ee1ff1c9)

```bash
git remote add origin [Git Repository URL]
```

## Local branch 변경

`.git/refs/remotes`에 있는 `tags` 폴더를 `.git/refs/tags`로 이동

`.git/refs/remotes`로 이동, `tags` 를 제외한 경로 내 파일들을 `.git/refs/heads`로 이동

```bash
# tags 폴더 이동
for t in $(git for-each-ref --format='%(refname:short)' refs/remotes/tags); do git tag ${t/tags\//} $t && git branch -D -r $t; done
# remotes 하위 폴더 heads로 이동
for b in $(git for-each-ref --format='%(refname:short)' refs/remotes); do git branch $b refs/remotes/$b && git branch -D -r $b; done

# trunk 브랜치 삭제
git branch -d trunk
```

## 확인 후 Push

```bash
# 브랜치 목록 확인
git branch -a

# 태그 목록 확인
git tag

# 브랜치별 소스 및 태그 푸시
git push origin --all
git push origin --tags
```

`GitLab` 의 프로젝트에 푸시를 하는 경우 브랜치명을 주의해야 한다. 브랜치 이름 생성 규칙에 의해 reject 가 발생할 수 있다. 브랜치 이름 생성 규칙은 "프로젝트 설정 > 저장소 > Push Rules > Branch name"에 정의되어 있다.
